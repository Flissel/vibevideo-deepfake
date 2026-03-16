"""
mouth_analyze.py - Per-frame mouth pixel diff time-series analysis.

Compares mouth region pixels frame-by-frame between original and lip-synced video.
Uses MediaPipe FaceLandmarker (478 landmarks) for precise mouth ROI extraction.

Outputs:
  - JSON with per-frame stats (mean_diff, max_diff, pct_changed, etc.)
  - Optional matplotlib plot of time series
  - Numpy .npz of per-frame diff maps (for adaptive_blend.py)

Usage:
  python mouth_analyze.py --original Surya.mp4 --synced lip_sync/Surya.mp4
  python mouth_analyze.py --all                    # all 4 lip-synced members
  python mouth_analyze.py --all --plot             # with matplotlib plots
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path(__file__).parent.parent.parent
ANALYSIS = VIDEO_DIR / "analysis.json"
OUT_DIR = VIDEO_DIR / "mouth_analysis"

# MediaPipe mouth outer contour landmarks (same as lip_sync.py)
MOUTH_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]

# Extended mouth region: include inner lips for better coverage
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
    """Extract mouth mask using MediaPipe landmarks.

    Returns:
        mask: float32 array (H, W) with values 0-1, or None if no face
        bbox: (y1, x1, y2, x2) bounding box of mouth region
    """
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    lms = result.face_landmarks[0]

    # Use both outer and inner mouth landmarks
    all_mouth = MOUTH_OUTER + MOUTH_INNER
    pts = np.array(
        [[int(lms[i].x * w), int(lms[i].y * h)] for i in all_mouth],
        dtype=np.int32
    )

    # Face height for adaptive expansion
    face_top_y = lms[10].y * h
    face_bot_y = lms[152].y * h
    face_height = max(face_bot_y - face_top_y, 50)

    expand_px = max(5, int(face_height * expand_ratio))

    # Create mask from convex hull
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    # Dilate to expand
    kernel_size = max(3, expand_px * 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Soft edges
    blur_r = max(5, int(face_height * 0.04))
    ksize = blur_r * 2 + 1
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), blur_r)

    # Bounding box of mask region
    ys, xs = np.where(mask > 128)
    if len(ys) == 0:
        return mask_f, None
    bbox = (ys.min(), xs.min(), ys.max(), xs.max())

    return mask_f, bbox


def analyze_pair(original_path, synced_path, name, plot=False, save_diffs=True):
    """Analyze mouth pixel differences between original and synced video.

    Returns dict with per-frame stats.
    """
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_sync = cv2.VideoCapture(str(synced_path))

    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sync = int(cap_sync.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_orig, total_sync)

    print(f"  Analyzing {name}: {total} frames ({total_orig} orig, {total_sync} sync)")

    landmarker = create_face_landmarker()

    frame_stats = []
    diff_maps = []  # store per-frame diff maps for adaptive blending

    for i in range(total):
        ret_o, frame_orig = cap_orig.read()
        ret_s, frame_sync = cap_sync.read()

        if not ret_o or not ret_s:
            break

        # Resize if needed
        h, w = frame_sync.shape[:2]
        if frame_orig.shape[:2] != (h, w):
            frame_orig = cv2.resize(frame_orig, (w, h))

        # Get mouth mask from original (more reliable face detection)
        frame_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        mask, bbox = get_mouth_mask(frame_rgb, landmarker)

        if mask is None or bbox is None:
            frame_stats.append({
                "frame": i,
                "time": i / fps,
                "mean_diff": 0,
                "max_diff": 0,
                "std_diff": 0,
                "pct_changed": 0,
                "no_face": True,
            })
            diff_maps.append(np.zeros((h, w), dtype=np.float32))
            continue

        # Per-pixel L2 norm difference across RGB channels in mouth region
        orig_f = frame_orig.astype(np.float32)
        sync_f = frame_sync.astype(np.float32)
        pixel_diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))  # (H, W)

        # Mask the diff to mouth region only
        mouth_diff = pixel_diff * (mask > 0.3).astype(np.float32)

        # Stats within mouth region
        mouth_pixels = pixel_diff[mask > 0.3]
        if len(mouth_pixels) == 0:
            frame_stats.append({
                "frame": i, "time": i / fps,
                "mean_diff": 0, "max_diff": 0, "std_diff": 0,
                "pct_changed": 0, "no_face": True,
            })
            diff_maps.append(np.zeros((h, w), dtype=np.float32))
            continue

        change_threshold = 15  # pixels differing by more than this are "changed"
        pct_changed = (mouth_pixels > change_threshold).sum() / len(mouth_pixels) * 100

        stats = {
            "frame": i,
            "time": round(i / fps, 3),
            "mean_diff": round(float(mouth_pixels.mean()), 2),
            "max_diff": round(float(mouth_pixels.max()), 2),
            "std_diff": round(float(mouth_pixels.std()), 2),
            "pct_changed": round(float(pct_changed), 1),
            "n_mouth_pixels": int(len(mouth_pixels)),
        }
        frame_stats.append(stats)
        diff_maps.append(pixel_diff)  # full frame diff for blending

        if (i + 1) % 30 == 0:
            print(f"    {i + 1}/{total} | mean={stats['mean_diff']:.1f} "
                  f"pct_changed={stats['pct_changed']:.1f}%", end="\r")

    cap_orig.release()
    cap_sync.release()
    landmarker.close()

    print(f"\n  Done: {len(frame_stats)} frames analyzed")

    # Summary stats
    means = [s["mean_diff"] for s in frame_stats if not s.get("no_face")]
    pcts = [s["pct_changed"] for s in frame_stats if not s.get("no_face")]

    summary = {
        "name": name,
        "total_frames": len(frame_stats),
        "fps": fps,
        "avg_mean_diff": round(np.mean(means), 2) if means else 0,
        "avg_pct_changed": round(np.mean(pcts), 1) if pcts else 0,
        "max_mean_diff": round(max(means), 2) if means else 0,
        "frames_with_face": sum(1 for s in frame_stats if not s.get("no_face")),
    }

    print(f"  Summary: avg_diff={summary['avg_mean_diff']}, "
          f"avg_pct_changed={summary['avg_pct_changed']}%, "
          f"max_diff={summary['max_mean_diff']}")

    # Save outputs
    OUT_DIR.mkdir(exist_ok=True)

    # JSON stats
    json_path = OUT_DIR / f"{name}_stats.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "frames": frame_stats}, f, indent=2)
    print(f"  Stats: {json_path}")

    # Numpy diff maps for adaptive blending
    if save_diffs:
        npz_path = OUT_DIR / f"{name}_diffs.npz"
        np.savez_compressed(npz_path, diffs=np.array(diff_maps, dtype=np.float32))
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"  Diffs: {npz_path} ({size_mb:.1f} MB)")

    # Plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

            times = [s["time"] for s in frame_stats]
            mean_diffs = [s["mean_diff"] for s in frame_stats]
            pct_changeds = [s["pct_changed"] for s in frame_stats]

            ax1.plot(times, mean_diffs, "b-", alpha=0.7, linewidth=0.8)
            ax1.fill_between(times, mean_diffs, alpha=0.2, color="blue")
            ax1.set_ylabel("Mean Pixel Diff (L2 norm)")
            ax1.set_title(f"{name}: Mouth Pixel Difference Over Time")
            ax1.grid(True, alpha=0.3)

            ax2.plot(times, pct_changeds, "r-", alpha=0.7, linewidth=0.8)
            ax2.fill_between(times, pct_changeds, alpha=0.2, color="red")
            ax2.set_ylabel("% Pixels Changed (>15)")
            ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = OUT_DIR / f"{name}_plot.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot: {plot_path}")
        except ImportError:
            print("  matplotlib not available, skipping plot")

    return summary


def load_analysis():
    with open(ANALYSIS, encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description="Per-frame mouth pixel diff time-series analysis"
    )
    p.add_argument("--original", help="Original video path")
    p.add_argument("--synced", help="Lip-synced video path")
    p.add_argument("--name", help="Person name (for output files)")
    p.add_argument("--all", action="store_true", help="Process all lip-synced members")
    p.add_argument("--only", help="Process only this person")
    p.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    p.add_argument("--no-diffs", action="store_true",
                   help="Skip saving diff maps (saves disk space)")
    args = p.parse_args()

    if args.all or args.only:
        entries = load_analysis()
        summaries = []

        for entry in entries:
            if entry.get("skip_lipsync"):
                continue
            name = entry["name"]
            if args.only and name.lower() != args.only.lower():
                continue

            # Original: extract clip range
            orig_path = VIDEO_DIR / entry["file"]
            sync_path = VIDEO_DIR / "lip_sync" / f"{name}.mp4"

            if not sync_path.exists():
                print(f"  SKIP {name}: {sync_path} not found")
                continue

            # We need the clipped original (same frame range as synced)
            # Extract clip to temp file
            import tempfile
            import subprocess
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            clip_start = entry.get("clip_start", 0)
            clip_end = entry.get("clip_end", entry["duration"])

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_clip = tmp.name

            subprocess.run([
                ffmpeg, "-y",
                "-i", str(orig_path),
                "-ss", str(clip_start),
                "-t", str(clip_end - clip_start),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an",
                str(tmp_clip),
            ], capture_output=True, check=True)

            summary = analyze_pair(
                tmp_clip, sync_path, name,
                plot=args.plot,
                save_diffs=not args.no_diffs,
            )
            summaries.append(summary)

            Path(tmp_clip).unlink(missing_ok=True)

        # Print final comparison
        if summaries:
            print(f"\n{'='*60}")
            print(f"  {'Name':<12} {'Avg Diff':>10} {'Avg %Changed':>14} {'Max Diff':>10}")
            print(f"  {'-'*46}")
            for s in summaries:
                print(f"  {s['name']:<12} {s['avg_mean_diff']:>10.2f} "
                      f"{s['avg_pct_changed']:>13.1f}% {s['max_mean_diff']:>10.2f}")
    else:
        if not args.original or not args.synced:
            p.error("Either --all or --original + --synced required")
        name = args.name or Path(args.original).stem
        analyze_pair(args.original, args.synced, name,
                     plot=args.plot, save_diffs=not args.no_diffs)


if __name__ == "__main__":
    main()
