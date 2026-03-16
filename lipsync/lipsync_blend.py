"""
lipsync_blend.py — Post-process MuseTalk lip-sync to flatten visible seams.

Combines three techniques (all adjustable):
  A) Poisson blending (cv2.seamlessClone) — eliminates color/lighting seams
  B) Adjustable feather blur — wider Gaussian on the blend mask
  C) Vertical bbox shift — move the blend region up/down to cover chin/jaw
  + LAB color correction to match skin tones

Usage:
  python lipsync_blend.py --original "orig.mp4" --lipsync "lip.mp4" -o "out.mp4"
  python lipsync_blend.py --original "orig.mp4" --lipsync "lip.mp4" --preview
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ── Face detection via OpenCV Haar cascade ──────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Cache last known face box for frames where detection fails
_last_face_box = None


def detect_face(frame):
    """Detect face and return (x, y, w, h) or cached result."""
    global _last_face_box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        # Pick largest face
        areas = [w * h for (_, _, w, h) in faces]
        _last_face_box = faces[np.argmax(areas)]
        return _last_face_box
    return _last_face_box  # use cached


def get_mouth_mask(frame, feather_radius=25, vertical_shift=0, expand_ratio=1.3):
    """Estimate mouth/lower-face region from face detection, return soft mask + center."""
    h, w = frame.shape[:2]
    face = detect_face(frame)
    if face is None:
        return None, None

    fx, fy, fw, fh = face

    # Mouth region = lower 45% of face box
    mouth_top = fy + int(fh * 0.55)
    mouth_bottom = fy + fh + int(fh * 0.1)  # extend slightly below face box
    mouth_left = fx + int(fw * 0.15)
    mouth_right = fx + fw - int(fw * 0.15)

    # Center of mouth region
    cx = (mouth_left + mouth_right) // 2
    cy = (mouth_top + mouth_bottom) // 2 + vertical_shift

    # Expand
    half_w = int((mouth_right - mouth_left) * expand_ratio) // 2
    half_h = int((mouth_bottom - mouth_top) * expand_ratio) // 2

    # Clamp
    cx = max(half_w, min(w - half_w, cx))
    cy = max(half_h, min(h - half_h, cy))

    # Create elliptical mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (half_w, half_h), 0, 0, 360, 255, -1)

    # Feather edges
    k = feather_radius * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask, (cx, cy)


def color_transfer_mouth(original, lipsync, mask):
    """Match color/brightness of lipsync mouth region to original (LAB space)."""
    mask_bool = mask > 30
    if not mask_bool.any():
        return lipsync

    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float64)
    lip_lab = cv2.cvtColor(lipsync, cv2.COLOR_BGR2LAB).astype(np.float64)

    for ch in range(3):
        orig_ch = orig_lab[:, :, ch][mask_bool]
        lip_ch = lip_lab[:, :, ch][mask_bool]
        if len(orig_ch) == 0:
            continue
        o_mean, o_std = orig_ch.mean(), max(orig_ch.std(), 1e-6)
        l_mean, l_std = lip_ch.mean(), max(lip_ch.std(), 1e-6)
        lip_lab[:, :, ch][mask_bool] = (lip_ch - l_mean) * (o_std / l_std) + o_mean

    lip_lab = np.clip(lip_lab, 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(lip_lab, cv2.COLOR_LAB2BGR)

    mask_3 = np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0
    return (corrected * mask_3 + lipsync * (1.0 - mask_3)).astype(np.uint8)


def alpha_blend(original, lipsync, mask, strength=1.0):
    """Simple alpha blend using soft mask."""
    alpha = (mask.astype(np.float32) / 255.0) * strength
    alpha = np.stack([alpha] * 3, axis=-1)
    return (lipsync * alpha + original * (1.0 - alpha)).astype(np.uint8)


def blend_frame(original, lipsync,
                feather_radius=25, vertical_shift=0, expand_ratio=1.3,
                poisson=True, color_correct=True, blend_strength=1.0):
    """Blend a single frame: lipsync mouth onto original body."""
    mask, center = get_mouth_mask(
        original,
        feather_radius=feather_radius,
        vertical_shift=vertical_shift,
        expand_ratio=expand_ratio,
    )
    if mask is None:
        return lipsync

    # Step 1: Color-correct lipsync mouth to match original skin tones
    if color_correct:
        lipsync = color_transfer_mouth(original, lipsync, mask)

    # Step 2: Poisson blending for invisible seams
    if poisson and center is not None:
        try:
            poisson_mask = (mask > 30).astype(np.uint8) * 255
            blended = cv2.seamlessClone(
                lipsync, original, poisson_mask, center, cv2.MIXED_CLONE
            )
        except cv2.error:
            blended = alpha_blend(original, lipsync, mask, blend_strength)
    else:
        blended = alpha_blend(original, lipsync, mask, blend_strength)

    return blended


def process_video(original_path, lipsync_path, output_path, **kwargs):
    """Process full video with blending."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cap_orig = cv2.VideoCapture(str(original_path))
    cap_lip = cv2.VideoCapture(str(lipsync_path))

    fps = cap_lip.get(cv2.CAP_PROP_FPS)
    w = int(cap_lip.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_lip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_lip.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_video = str(output_path) + "_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    global _last_face_box
    _last_face_box = None

    frame_idx = 0
    print(f"Processing {total} frames...")
    while True:
        ret_o, frame_orig = cap_orig.read()
        ret_l, frame_lip = cap_lip.read()

        if not ret_l:
            break
        if not ret_o:
            writer.write(frame_lip)
            frame_idx += 1
            continue

        if frame_orig.shape[:2] != frame_lip.shape[:2]:
            frame_orig = cv2.resize(frame_orig, (w, h))

        blended = blend_frame(frame_orig, frame_lip, **kwargs)
        writer.write(blended)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  {frame_idx}/{total}", end="\r")

    writer.release()
    cap_orig.release()
    cap_lip.release()
    print(f"\n  {frame_idx}/{total} frames done.")

    # Mux audio from lipsync video
    output_final = str(output_path)
    subprocess.run([
        ffmpeg, "-y",
        "-i", tmp_video,
        "-i", str(lipsync_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart",
        "-shortest",
        output_final,
    ], capture_output=True, check=True)

    Path(tmp_video).unlink(missing_ok=True)
    size_mb = Path(output_final).stat().st_size / (1024 * 1024)
    print(f"Output: {output_final} ({size_mb:.1f} MB)")


def preview_frame(original_path, lipsync_path, output_path="preview_blend.png",
                  frame_num=15, **kwargs):
    """Save a single blended frame for quick preview (original | lipsync | blended)."""
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_lip = cv2.VideoCapture(str(lipsync_path))

    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    cap_lip.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    _, frame_orig = cap_orig.read()
    _, frame_lip = cap_lip.read()
    cap_orig.release()
    cap_lip.release()

    if frame_orig is None or frame_lip is None:
        print("Could not read frames.")
        return

    w, h = frame_lip.shape[1], frame_lip.shape[0]
    if frame_orig.shape[:2] != (h, w):
        frame_orig = cv2.resize(frame_orig, (w, h))

    blended = blend_frame(frame_orig, frame_lip, **kwargs)

    # Side-by-side comparison
    comparison = np.hstack([frame_orig, frame_lip, blended])
    cv2.imwrite(str(output_path), comparison)
    print(f"Preview saved: {output_path}  (original | lipsync | blended)")


def main():
    p = argparse.ArgumentParser(
        description="Post-process MuseTalk lip-sync to flatten visible seams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Adjustable parameters:
  --feather   Blur radius for mask edges (higher = softer transition)
  --shift     Move blend region up(-) or down(+) in pixels
  --expand    Scale the mouth region (1.0 = tight, 2.0 = very wide)
  --strength  Alpha blend strength (0.0 = original, 1.0 = full lipsync)

Examples:
  # Preview first:
  python lipsync_blend.py --original orig.mp4 --lipsync lip.mp4 --preview

  # Default (Poisson + color correction + feather):
  python lipsync_blend.py --original orig.mp4 --lipsync lip.mp4 -o out.mp4

  # Softer blend, shift down 10px:
  python lipsync_blend.py ... --feather 40 --shift 10 --expand 1.5

  # Alpha blend only (no Poisson):
  python lipsync_blend.py ... --no-poisson --strength 0.85
        """,
    )
    p.add_argument("--original", required=True, help="Original video (before lip-sync)")
    p.add_argument("--lipsync", required=True, help="MuseTalk lip-synced video")
    p.add_argument("-o", "--output", default="lip_sync/blended.mp4", help="Output path")
    p.add_argument("--preview", action="store_true", help="Save single frame comparison")
    p.add_argument("--frame", type=int, default=15, help="Frame number for preview")

    # Adjustable parameters
    p.add_argument("--feather", type=int, default=25,
                   help="Feather blur radius (default: 25, higher=softer)")
    p.add_argument("--shift", type=int, default=0,
                   help="Vertical shift in px (+ = down, - = up)")
    p.add_argument("--expand", type=float, default=1.3,
                   help="Mouth region expand ratio (default: 1.3)")
    p.add_argument("--no-poisson", action="store_true",
                   help="Disable Poisson blending, use alpha blend")
    p.add_argument("--no-color-correct", action="store_true",
                   help="Disable LAB color correction")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Blend strength 0-1 (alpha mode only)")

    args = p.parse_args()

    kwargs = dict(
        feather_radius=args.feather,
        vertical_shift=args.shift,
        expand_ratio=args.expand,
        poisson=not args.no_poisson,
        color_correct=not args.no_color_correct,
        blend_strength=args.strength,
    )

    print(f"Settings: feather={args.feather}, shift={args.shift}, "
          f"expand={args.expand}, poisson={not args.no_poisson}, "
          f"color={not args.no_color_correct}, strength={args.strength}")

    if args.preview:
        preview_frame(args.original, args.lipsync, frame_num=args.frame, **kwargs)
    else:
        process_video(args.original, args.lipsync, args.output, **kwargs)


if __name__ == "__main__":
    main()
