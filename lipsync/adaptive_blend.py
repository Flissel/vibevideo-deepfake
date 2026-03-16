"""
adaptive_blend.py - Diff-driven per-pixel adaptive blending.

Uses per-frame pixel differences between original and MuseTalk output to create
a smart blend mask: where pixels barely changed -> use original (natural texture),
where they changed a lot -> use MuseTalk (active lip sync).

This replaces the LS boundary blend with a data-driven approach that preserves
original skin/lip texture wherever possible.

Pipeline:
  1. Read original clip + MuseTalk output (or current lip_sync file)
  2. Per frame: compute pixel diff, create adaptive alpha mask
  3. Temporal + spatial smoothing on alpha to prevent flicker
  4. Blend: result = synced * alpha + original * (1 - alpha)
  5. Output final video with audio from synced

Usage:
  python adaptive_blend.py --original clip.mp4 --synced musetalk.mp4 -o output.mp4
  python adaptive_blend.py --all                           # all 4 members
  python adaptive_blend.py --all --threshold-low 8 --threshold-high 35
"""
import argparse
import json
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path(__file__).parent.parent
ANALYSIS = VIDEO_DIR / "analysis.json"
OUT_DIR = VIDEO_DIR / "lip_sync"

# MediaPipe mouth landmarks (reuse from lip_sync.py / mouth_analyze.py)
MOUTH_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
MOUTH_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]


def create_face_landmarker():
    """Create MediaPipe FaceLandmarker."""
    import mediapipe as mp
    model_path = VIDEO_DIR / "face_landmarker.task"
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def get_mouth_mask(frame_rgb, landmarker, expand_ratio=0.08):
    """Extract soft mouth mask using MediaPipe landmarks.

    Returns float32 (H, W) with values 0-1, or None.
    """
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    all_mouth = MOUTH_OUTER + MOUTH_INNER
    pts = np.array(
        [[int(lms[i].x * w), int(lms[i].y * h)] for i in all_mouth],
        dtype=np.int32,
    )

    face_top_y = lms[10].y * h
    face_bot_y = lms[152].y * h
    face_height = max(face_bot_y - face_top_y, 50)
    expand_px = max(5, int(face_height * expand_ratio))

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    kernel_size = max(3, expand_px * 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)

    blur_r = max(5, int(face_height * 0.05))
    ksize = blur_r * 2 + 1
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), blur_r)

    return mask_f


def adaptive_blend_frame(original, synced, mouth_mask,
                         threshold_low=10, threshold_high=40,
                         max_alpha=1.0):
    """Create per-pixel adaptive blend based on difference magnitude.

    Where pixels barely changed: use original (preserves texture).
    Where they changed a lot: use synced (active lip movement).

    Args:
        max_alpha: cap on blend weight (0.5 = always keep 50% original texture)

    Returns:
        blended frame (uint8), alpha map (float32)
    """
    if mouth_mask is None:
        return synced, np.zeros(synced.shape[:2], dtype=np.float32)

    orig_f = original.astype(np.float32)
    sync_f = synced.astype(np.float32)

    # Per-pixel L2 distance across BGR channels
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))

    # Smooth ramp: low diff -> 0 (use original), high diff -> max_alpha (use synced)
    alpha = np.clip(
        (diff - threshold_low) / max(threshold_high - threshold_low, 1),
        0.0, 1.0,
    ).astype(np.float32) * max_alpha

    # Only apply within mouth mask
    alpha = alpha * mouth_mask

    # Spatial smoothing for soft transitions
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)

    # Blend
    alpha3 = alpha[..., np.newaxis]
    blended = (sync_f * alpha3 + orig_f * (1.0 - alpha3)).astype(np.uint8)

    return blended, alpha


def process_video(original_path, synced_path, output_path,
                  threshold_low=10, threshold_high=40,
                  temporal_window=5, max_alpha=1.0):
    """Process full video with adaptive blending.

    Args:
        temporal_window: number of frames for rolling alpha average (flicker prevention)
    """
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cap_orig = cv2.VideoCapture(str(original_path))
    cap_sync = cv2.VideoCapture(str(synced_path))

    fps = cap_sync.get(cv2.CAP_PROP_FPS)
    w = int(cap_sync.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_sync.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(
        int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_sync.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    print(f"  Adaptive blend: {total} frames, thresholds=({threshold_low}, {threshold_high}), "
          f"temporal_window={temporal_window}, max_alpha={max_alpha}")

    landmarker = create_face_landmarker()

    # Two-pass approach for temporal smoothing:
    # Pass 1: compute all alpha maps
    # Pass 2: apply temporally smoothed alpha
    # (or single-pass with rolling buffer)

    # Single-pass with causal rolling buffer
    tmp_video = str(output_path) + "_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    alpha_buffer = deque(maxlen=temporal_window)
    # We need lookahead for centered smoothing, so buffer frames too
    frame_buffer = deque(maxlen=temporal_window)
    orig_buffer = deque(maxlen=temporal_window)
    sync_buffer = deque(maxlen=temporal_window)

    half_win = temporal_window // 2
    frames_read = 0
    frames_written = 0

    # Read all frames into alpha maps first, then blend with smoothed alpha
    # For memory: do two-pass — first pass computes alphas, second pass blends

    # Actually, let's do a single-pass with centered temporal smoothing:
    # buffer half_win frames, then start writing from the middle

    all_alphas = []
    all_orig = []
    all_sync = []

    print(f"  Pass 1: computing per-frame alpha maps...")
    for i in range(total):
        ret_o, frame_orig = cap_orig.read()
        ret_s, frame_sync = cap_sync.read()

        if not ret_o or not ret_s:
            break

        if frame_orig.shape[:2] != (h, w):
            frame_orig = cv2.resize(frame_orig, (w, h))

        frame_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        mouth_mask = get_mouth_mask(frame_rgb, landmarker)

        _, alpha = adaptive_blend_frame(
            frame_orig, frame_sync, mouth_mask,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            max_alpha=max_alpha,
        )

        all_alphas.append(alpha)
        all_orig.append(frame_orig)
        all_sync.append(frame_sync)

        if (i + 1) % 30 == 0:
            pct = (i + 1) / total * 100
            print(f"    {i + 1}/{total} ({pct:.0f}%)", end="\r")

    cap_orig.release()
    cap_sync.release()
    landmarker.close()

    n_frames = len(all_alphas)
    print(f"\n  Pass 2: temporal smoothing + blending {n_frames} frames...")

    for i in range(n_frames):
        # Centered temporal smoothing
        start = max(0, i - half_win)
        end = min(n_frames, i + half_win + 1)
        smoothed_alpha = np.mean(all_alphas[start:end], axis=0).astype(np.float32)

        # Re-apply spatial blur after temporal smoothing
        smoothed_alpha = cv2.GaussianBlur(smoothed_alpha, (9, 9), 0)

        # Blend
        alpha3 = smoothed_alpha[..., np.newaxis]
        orig_f = all_orig[i].astype(np.float32)
        sync_f = all_sync[i].astype(np.float32)
        blended = (sync_f * alpha3 + orig_f * (1.0 - alpha3)).astype(np.uint8)

        writer.write(blended)
        frames_written += 1

        if (i + 1) % 30 == 0:
            pct = (i + 1) / n_frames * 100
            avg_alpha = smoothed_alpha.max()
            print(f"    {i + 1}/{n_frames} ({pct:.0f}%) max_alpha={avg_alpha:.2f}", end="\r")

    writer.release()
    print(f"\n  Written {frames_written} frames")

    # Free memory
    del all_alphas, all_orig, all_sync

    # Mux audio from synced video
    # Use a separate temp output to avoid reading+writing the same file
    output_final = str(output_path)
    tmp_muxed = output_final + "_muxed.mp4"
    subprocess.run([
        ffmpeg, "-y",
        "-i", tmp_video,
        "-i", str(synced_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart",
        "-shortest",
        tmp_muxed,
    ], capture_output=True, check=True)

    Path(tmp_video).unlink(missing_ok=True)
    # Replace original with muxed result
    Path(tmp_muxed).replace(output_final)
    size_mb = Path(output_final).stat().st_size / (1024 * 1024)
    print(f"  Output: {output_final} ({size_mb:.1f} MB)")
    return Path(output_final)


def load_analysis():
    with open(ANALYSIS, encoding="utf-8") as f:
        return json.load(f)


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    p = argparse.ArgumentParser(
        description="Diff-driven per-pixel adaptive blending for lip-sync"
    )
    p.add_argument("--original", help="Original video path")
    p.add_argument("--synced", help="Lip-synced video path")
    p.add_argument("-o", "--output", help="Output path")
    p.add_argument("--all", action="store_true", help="Process all lip-synced members")
    p.add_argument("--only", help="Process only this person")
    p.add_argument("--threshold-low", type=float, default=10,
                   help="Below this diff: use original (default: 10)")
    p.add_argument("--threshold-high", type=float, default=40,
                   help="Above this diff: use synced (default: 40)")
    p.add_argument("--temporal-window", type=int, default=5,
                   help="Frames for temporal alpha smoothing (default: 5)")
    p.add_argument("--max-alpha", type=float, default=1.0,
                   help="Max blend alpha cap (0.5 = always keep 50%% original, default: 1.0)")
    p.add_argument("--musetalk-raw", action="store_true",
                   help="Re-run MuseTalk to get raw output (skip LS blend)")
    args = p.parse_args()

    if args.all or args.only:
        entries = load_analysis()

        for entry in entries:
            if entry.get("skip_lipsync"):
                continue
            name = entry["name"]
            if args.only and name.lower() != args.only.lower():
                continue

            print(f"\n{'='*60}")
            print(f"  {name}: Adaptive Blend")
            print(f"{'='*60}")

            orig_path = VIDEO_DIR / entry["file"]
            clip_start = entry.get("clip_start", 0)
            clip_end = entry.get("clip_end", entry["duration"])

            # Extract original clip
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_clip = tmp.name

            ffmpeg = get_ffmpeg()
            subprocess.run([
                ffmpeg, "-y",
                "-i", str(orig_path),
                "-ss", str(clip_start),
                "-t", str(clip_end - clip_start),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an",
                str(tmp_clip),
            ], capture_output=True, check=True)

            if args.musetalk_raw:
                # Re-run MuseTalk to get raw output (without LS blend)
                print(f"  Re-running MuseTalk for raw output...")
                from musetalk_lipsync import run_musetalk
                from align_audio import align_tts_to_original

                tts_path = VIDEO_DIR / "tts" / f"{name}.mp3"
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_a:
                    aligned_path = tmp_a.name

                # Align audio
                align_tts_to_original(
                    original_path=orig_path,
                    tts_path=tts_path,
                    output_path=aligned_path,
                    clip_start=clip_start,
                    clip_end=clip_end,
                )

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_m:
                    musetalk_out = tmp_m.name

                run_musetalk(
                    video_path=tmp_clip,
                    audio_path=aligned_path,
                    output_path=musetalk_out,
                    version="v15",
                    use_float16=True,
                    batch_size=8,
                )

                synced_path = musetalk_out
                Path(aligned_path).unlink(missing_ok=True)
            else:
                # Use existing lip_sync output (which includes LS blend)
                # Copy to temp since output overwrites the same file
                orig_synced = VIDEO_DIR / "lip_sync" / f"{name}.mp4"
                if not orig_synced.exists():
                    print(f"  SKIP {name}: {orig_synced} not found")
                    Path(tmp_clip).unlink(missing_ok=True)
                    continue
                import shutil
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_s:
                    synced_copy = tmp_s.name
                shutil.copy2(str(orig_synced), synced_copy)
                synced_path = synced_copy

            output = OUT_DIR / f"{name}.mp4"
            process_video(
                tmp_clip, synced_path, output,
                threshold_low=args.threshold_low,
                threshold_high=args.threshold_high,
                temporal_window=args.temporal_window,
                max_alpha=args.max_alpha,
            )

            Path(tmp_clip).unlink(missing_ok=True)
            if args.musetalk_raw:
                Path(musetalk_out).unlink(missing_ok=True)
            elif synced_path != str(orig_synced):
                Path(synced_path).unlink(missing_ok=True)

        print(f"\n{'='*60}")
        print(f"  All done!")
        print(f"{'='*60}")

    else:
        if not args.original or not args.synced:
            p.error("Either --all or --original + --synced required")
        output = args.output or "adaptive_blended.mp4"
        process_video(
            args.original, args.synced, output,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            temporal_window=args.temporal_window,
            max_alpha=args.max_alpha,
        )


if __name__ == "__main__":
    main()
