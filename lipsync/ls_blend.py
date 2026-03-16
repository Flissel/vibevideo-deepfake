"""
ls_blend.py - Least-squares boundary matching post-processor.

Per-frame automatic seam elimination at mask boundary. Samples pixels on both
sides of the mask edge, fits per-channel affine color transform using
np.linalg.lstsq, then soft-blends the corrected generated face onto the original.

Usage:
  python ls_blend.py --original orig.mp4 --generated musetalk.mp4 -o output.mp4
  python ls_blend.py --original orig.mp4 --generated musetalk.mp4 --preview
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


# Face detection for mouth mask
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_last_face_box = None


def detect_face(frame):
    """Detect face, return (x, y, w, h) or cached result."""
    global _last_face_box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        areas = [w * h for (_, _, w, h) in faces]
        _last_face_box = faces[np.argmax(areas)]
        return _last_face_box
    return _last_face_box


def get_face_mask(frame, expand_ratio=1.4, feather_radius=15):
    """Get soft mask covering the lower face (mouth + jaw region)."""
    h, w = frame.shape[:2]
    face = detect_face(frame)
    if face is None:
        return None

    fx, fy, fw, fh = face

    # Mouth/lower face region = lower 50% of face box
    mouth_top = fy + int(fh * 0.50)
    mouth_bottom = fy + fh + int(fh * 0.15)
    mouth_left = fx + int(fw * 0.10)
    mouth_right = fx + fw - int(fw * 0.10)

    cx = (mouth_left + mouth_right) // 2
    cy = (mouth_top + mouth_bottom) // 2

    half_w = int((mouth_right - mouth_left) * expand_ratio) // 2
    half_h = int((mouth_bottom - mouth_top) * expand_ratio) // 2

    cx = max(half_w, min(w - half_w, cx))
    cy = max(half_h, min(h - half_h, cy))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (half_w, half_h), 0, 0, 360, 255, -1)

    k = feather_radius * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def ls_boundary_match(original, generated, mask, band_width=7,
                      min_boundary_pixels=50):
    """Per-frame least-squares color correction at mask boundary.

    1. Find boundary band (dilate - erode) on both sides
    2. Sample boundary pixels from both sides
    3. Fit per-channel affine: corrected = a * gen + b
    4. Apply correction to generated pixels inside mask
    5. Soft-blend with mask
    """
    if mask is None:
        return generated

    mask_binary = (mask > 128).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (band_width, band_width))

    # Outer boundary band: pixels just outside the mask
    dilated = cv2.dilate(mask_binary, kernel)
    outer_band = dilated - mask_binary  # ring outside mask

    # Inner boundary band: pixels just inside the mask
    eroded = cv2.erode(mask_binary, kernel)
    inner_band = mask_binary - eroded  # ring inside mask

    outer_mask = outer_band > 128
    inner_mask = inner_band > 128

    n_outer = outer_mask.sum()
    n_inner = inner_mask.sum()

    if n_outer < min_boundary_pixels or n_inner < min_boundary_pixels:
        # Not enough boundary pixels, fall back to simple blend
        alpha = mask[..., np.newaxis].astype(np.float32) / 255.0
        return (generated * alpha + original * (1.0 - alpha)).astype(np.uint8)

    # Sample boundary pixels
    orig_outer = original[outer_mask].astype(np.float64)   # ground truth outside
    orig_inner = original[inner_mask].astype(np.float64)   # ground truth inside
    gen_inner = generated[inner_mask].astype(np.float64)    # generated inside

    # For the correction target, we want generated-inside to look like
    # what original-inside looks like, using the outer boundary as reference.
    # We fit: orig_outer ≈ a * gen_at_outer + b  (what the transform should do)
    # But we only have gen_inner pixels, not gen_outer.
    # Instead: fit the transform so gen_inner maps to orig_inner
    # (match the generated boundary pixels to original boundary pixels)

    corrected = generated.copy().astype(np.float64)

    for ch in range(3):
        # Use inner boundary: generated_inner -> original_inner
        gi = gen_inner[:, ch]
        oi = orig_inner[:, ch]

        # Fit affine: oi ≈ a * gi + b
        A = np.column_stack([gi, np.ones(len(gi))])
        coeffs, _, _, _ = np.linalg.lstsq(A, oi, rcond=None)
        a, b = coeffs

        # Clamp extreme corrections
        a = np.clip(a, 0.5, 2.0)
        b = np.clip(b, -50, 50)

        # Apply to all pixels in the mask region
        mask_region = mask > 30
        corrected[:, :, ch][mask_region] = (
            a * corrected[:, :, ch][mask_region] + b
        )

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Soft blend using the feathered mask
    alpha = mask[..., np.newaxis].astype(np.float32) / 255.0
    result = (corrected * alpha + original * (1.0 - alpha)).astype(np.uint8)
    return result


def process_video(original_path, generated_path, output_path,
                  expand_ratio=1.4, feather_radius=15, band_width=7):
    """Process full video with LS boundary matching."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cap_orig = cv2.VideoCapture(str(original_path))
    cap_gen = cv2.VideoCapture(str(generated_path))

    fps = cap_gen.get(cv2.CAP_PROP_FPS)
    w = int(cap_gen.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_gen.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_video = str(output_path) + "_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    global _last_face_box
    _last_face_box = None

    frame_idx = 0
    print(f"  LS blending {total} frames...")
    while True:
        ret_o, frame_orig = cap_orig.read()
        ret_g, frame_gen = cap_gen.read()

        if not ret_g:
            break
        if not ret_o:
            writer.write(frame_gen)
            frame_idx += 1
            continue

        if frame_orig.shape[:2] != frame_gen.shape[:2]:
            frame_orig = cv2.resize(frame_orig, (w, h))

        mask = get_face_mask(frame_orig, expand_ratio=expand_ratio,
                             feather_radius=feather_radius)
        blended = ls_boundary_match(frame_orig, frame_gen, mask,
                                     band_width=band_width)
        writer.write(blended)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"    {frame_idx}/{total}", end="\r")

    writer.release()
    cap_orig.release()
    cap_gen.release()
    print(f"    {frame_idx}/{total} frames done.")

    # Mux audio from generated video
    output_final = str(output_path)
    subprocess.run([
        ffmpeg, "-y",
        "-i", tmp_video,
        "-i", str(generated_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart",
        "-shortest",
        output_final,
    ], capture_output=True, check=True)

    Path(tmp_video).unlink(missing_ok=True)
    size_mb = Path(output_final).stat().st_size / (1024 * 1024)
    print(f"  Output: {output_final} ({size_mb:.1f} MB)")
    return Path(output_final)


def preview_frame(original_path, generated_path, output_path="preview_ls.png",
                  frame_num=15, **kwargs):
    """Save side-by-side comparison: original | generated | LS-blended."""
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_gen = cv2.VideoCapture(str(generated_path))

    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    cap_gen.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    _, frame_orig = cap_orig.read()
    _, frame_gen = cap_gen.read()
    cap_orig.release()
    cap_gen.release()

    if frame_orig is None or frame_gen is None:
        print("Could not read frames.")
        return

    w, h = frame_gen.shape[1], frame_gen.shape[0]
    if frame_orig.shape[:2] != (h, w):
        frame_orig = cv2.resize(frame_orig, (w, h))

    mask = get_face_mask(frame_orig, **kwargs)
    blended = ls_boundary_match(frame_orig, frame_gen, mask)

    comparison = np.hstack([frame_orig, frame_gen, blended])
    cv2.imwrite(str(output_path), comparison)
    print(f"Preview saved: {output_path}  (original | generated | LS-blended)")


def main():
    p = argparse.ArgumentParser(
        description="Least-squares boundary matching post-processor"
    )
    p.add_argument("--original", required=True, help="Original video")
    p.add_argument("--generated", required=True, help="MuseTalk output video")
    p.add_argument("-o", "--output", default="lip_sync/ls_blended.mp4")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--frame", type=int, default=15)
    p.add_argument("--expand", type=float, default=1.4)
    p.add_argument("--feather", type=int, default=15)
    p.add_argument("--band-width", type=int, default=7)
    args = p.parse_args()

    kwargs = dict(expand_ratio=args.expand, feather_radius=args.feather)

    if args.preview:
        preview_frame(args.original, args.generated, frame_num=args.frame,
                      **kwargs)
    else:
        process_video(args.original, args.generated, args.output,
                      band_width=args.band_width, **kwargs)


if __name__ == "__main__":
    main()
