"""
temporal_judge.py - Stats-based temporal naturalness judge for lip-sync blending.

Learns what "natural" looks like from the original video only, then uses
that as a judge to score each synced frame's naturalness. Unnatural pixels
get reverted to original.

How it works:
  Phase 1 (Learn): For each mouth pixel in the original video, compute:
    - mean, std over time (expected color range)
    - temporal gradient std (expected motion speed)
    - autocorrelation at lag-1 (expected smoothness)

  Phase 2 (Judge): For each synced frame, score each pixel:
    - Color deviation: how far from original's mean (in std units)
    - Motion deviation: is frame-to-frame change abnormally fast?
    - Smoothness deviation: does autocorrelation pattern match?
    Combined into a single "unnaturalness" score per pixel.
    High score = artifact → use original. Low score = natural → keep synced.

  Phase 3 (Blend): unnaturalness score becomes the blend alpha:
    alpha=0 → use original (unnatural synced pixel)
    alpha=1 → use synced (looks natural)
    (Inverted from adaptive_blend: here low alpha = keep synced)

Usage:
  python temporal_judge.py --only Surya
  python temporal_judge.py --all
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from adaptive_blend import (
    create_face_landmarker,
    get_mouth_mask,
    VIDEO_DIR,
)

ANALYSIS = VIDEO_DIR / "analysis.json"
OUT_DIR = VIDEO_DIR / "test_surya"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_analysis():
    with open(ANALYSIS, encoding="utf-8") as f:
        return json.load(f)


def extract_clip(video_path, clip_start, clip_end, out_path):
    """Extract clip segment."""
    ffmpeg = get_ffmpeg()
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(video_path),
        "-ss", str(clip_start), "-t", str(clip_end - clip_start),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        str(out_path),
    ], capture_output=True, check=True)


def read_all_frames(video_path, max_frames=None):
    """Read all frames from a video as uint8 BGR arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps, w, h


def compute_mouth_masks(frames, landmarker):
    """Compute mouth masks for all frames."""
    masks = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = get_mouth_mask(rgb, landmarker)
        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=np.float32)
        masks.append(mask)
    return masks


def learn_temporal_stats(frames, masks):
    """Phase 1: Learn per-pixel temporal statistics from original video.

    Uses incremental (Welford's) computation to avoid allocating full stack.

    Returns dict with arrays shaped (H, W, 3) or (H, W):
        - pixel_mean: mean color per pixel over time
        - pixel_std: std of color per pixel over time
        - grad_std: std of frame-to-frame gradient per pixel
        - autocorr: lag-1 autocorrelation per pixel (smoothness)
        - mouth_frequency: how often each pixel is in the mouth mask
    """
    n = len(frames)
    h, w = frames[0].shape[:2]

    # Incremental mean + variance (Welford's algorithm)
    pixel_mean = np.zeros((h, w, 3), dtype=np.float64)
    pixel_m2 = np.zeros((h, w, 3), dtype=np.float64)

    grad_mean = np.zeros((h, w, 3), dtype=np.float64)
    grad_m2 = np.zeros((h, w, 3), dtype=np.float64)

    # For autocorrelation: E[x_t * x_{t+1}], E[x_t], E[x_t^2]
    cross_sum = np.zeros((h, w, 3), dtype=np.float64)
    mouth_freq = np.zeros((h, w), dtype=np.float64)

    prev_frame = None
    n_grads = 0

    for i, frame in enumerate(frames):
        f = frame.astype(np.float64)

        # Welford update for pixel stats
        delta = f - pixel_mean
        pixel_mean += delta / (i + 1)
        delta2 = f - pixel_mean
        pixel_m2 += delta * delta2

        # Gradient stats (frame-to-frame)
        if prev_frame is not None:
            grad = f - prev_frame
            n_grads += 1
            g_delta = grad - grad_mean
            grad_mean += g_delta / n_grads
            g_delta2 = grad - grad_mean
            grad_m2 += g_delta * g_delta2

        # Cross-product for autocorrelation
        if prev_frame is not None:
            cross_sum += prev_frame * f

        prev_frame = f.copy()
        mouth_freq += masks[i].astype(np.float64)

    # Finalize
    pixel_var = pixel_m2 / max(n - 1, 1)
    pixel_std = np.sqrt(pixel_var).astype(np.float32) + 1e-6

    grad_var = grad_m2 / max(n_grads - 1, 1)
    grad_std = np.sqrt(grad_var).astype(np.float32) + 1e-6

    # Autocorrelation: corr = (E[x_t * x_{t+1}] - mu^2) / var
    cross_mean = cross_sum / max(n - 1, 1)
    autocorr = (cross_mean - pixel_mean ** 2) / (pixel_var + 1e-6)
    autocorr = np.clip(autocorr, -1, 1).astype(np.float32)

    pixel_mean = pixel_mean.astype(np.float32)
    grad_mean = grad_mean.astype(np.float32)
    mouth_freq = (mouth_freq / n).astype(np.float32)

    print(f"    Learned stats from {n} frames")
    print(f"    Mouth region: {(mouth_freq > 0.5).sum()} pixels frequently in mouth")
    print(f"    Mean pixel_std in mouth: {pixel_std[mouth_freq > 0.5].mean():.1f}")
    print(f"    Mean grad_std in mouth: {grad_std[mouth_freq > 0.5].mean():.1f}")
    print(f"    Mean autocorr in mouth: {autocorr[mouth_freq > 0.5].mean():.3f}")

    return {
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "autocorr": autocorr,
        "mouth_freq": mouth_freq,
    }


def judge_frame(synced_frame, prev_synced_frame, stats, mouth_mask,
                w_color=0.4, w_motion=0.4, w_smooth=0.2):
    """Phase 2: Score each pixel's naturalness in the synced frame.

    Returns unnaturalness score (H, W), 0 = perfectly natural, high = artifact.
    """
    sf = synced_frame.astype(np.float32)

    # 1. Color deviation: how many stds away from original mean?
    color_dev = np.abs(sf - stats["pixel_mean"]) / stats["pixel_std"]
    color_score = color_dev.mean(axis=2)  # average across BGR channels

    # 2. Motion deviation: is frame-to-frame gradient abnormal?
    if prev_synced_frame is not None:
        pf = prev_synced_frame.astype(np.float32)
        grad = sf - pf
        motion_dev = np.abs(grad - stats["grad_mean"]) / stats["grad_std"]
        motion_score = motion_dev.mean(axis=2)
    else:
        motion_score = np.zeros_like(color_score)

    # 3. Smoothness: we can't compute autocorrelation from one frame,
    #    but we can check if the gradient direction is consistent with
    #    the original's autocorrelation pattern.
    #    High autocorrelation in original = pixel should change slowly.
    #    If synced pixel jumps a lot where original was smooth → unnatural.
    if prev_synced_frame is not None:
        grad = sf - pf
        # Where original had high autocorrelation (smooth), penalize large gradients
        orig_smoothness = stats["autocorr"].mean(axis=2)  # (H, W)
        # smoothness_penalty: high when original is smooth but synced jumps
        grad_mag = np.sqrt(np.sum(grad ** 2, axis=2))
        expected_grad = np.sqrt(np.sum(stats["grad_std"] ** 2, axis=2))
        smooth_score = orig_smoothness * (grad_mag / (expected_grad + 1e-6))
    else:
        smooth_score = np.zeros_like(color_score)

    # Combined unnaturalness score
    unnaturalness = (w_color * color_score +
                     w_motion * motion_score +
                     w_smooth * smooth_score)

    # Apply mouth mask — only judge mouth region
    unnaturalness = unnaturalness * mouth_mask

    return unnaturalness


def unnaturalness_to_alpha(unnaturalness, low=1.0, high=3.0, max_alpha=0.8):
    """Convert unnaturalness score to blend alpha.

    INVERTED logic:
      low unnaturalness (natural) → high alpha → use synced
      high unnaturalness (artifact) → low alpha → use original

    Args:
        low: below this score, pixel is considered natural (alpha=max_alpha)
        high: above this score, pixel is considered artifact (alpha=0)
        max_alpha: cap on blend weight (never go 100% synced)
    """
    # Linear ramp: high unnaturalness → 0, low → max_alpha
    alpha = np.clip(
        (high - unnaturalness) / max(high - low, 0.01),
        0.0, 1.0
    ).astype(np.float32) * max_alpha

    return alpha


def process_person(entry, output_suffix="judge",
                   temporal_window=7, max_alpha=0.8,
                   thresh_low=0.3, thresh_high=1.0):
    """Full pipeline for one person."""
    t0 = time.time()
    name = entry["name"]
    ffmpeg = get_ffmpeg()
    OUT_DIR.mkdir(exist_ok=True)

    clip_start = entry.get("clip_start", 0)
    clip_end = entry.get("clip_end", entry["duration"])

    # Extract original clip
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_clip = tmp.name
    extract_clip(VIDEO_DIR / entry["file"], clip_start, clip_end, tmp_clip)

    synced_path = VIDEO_DIR / "lip_sync" / f"{name}.mp4"
    if not synced_path.exists():
        print(f"  SKIP: {synced_path} not found")
        Path(tmp_clip).unlink(missing_ok=True)
        return None

    print(f"  Reading frames...")
    orig_frames, fps, vid_w, vid_h = read_all_frames(tmp_clip)
    sync_frames, _, _, _ = read_all_frames(synced_path)
    n_frames = min(len(orig_frames), len(sync_frames))
    orig_frames = orig_frames[:n_frames]
    sync_frames = sync_frames[:n_frames]

    # Resize original if needed
    for i in range(n_frames):
        if orig_frames[i].shape[:2] != (vid_h, vid_w):
            orig_frames[i] = cv2.resize(orig_frames[i], (vid_w, vid_h))

    print(f"  Computing mouth masks ({n_frames} frames)...")
    landmarker = create_face_landmarker()
    orig_masks = compute_mouth_masks(orig_frames, landmarker)
    landmarker.close()

    # Phase 1: Learn from original
    print(f"  Phase 1: Learning temporal stats from original...")
    stats = learn_temporal_stats(orig_frames, orig_masks)

    # Phase 2 + 3: Judge synced frames and compute alpha
    print(f"  Phase 2: Judging synced frames...")
    all_alphas = []
    all_scores = []

    for i in range(n_frames):
        prev_sync = sync_frames[i - 1] if i > 0 else None
        score = judge_frame(
            sync_frames[i], prev_sync, stats, orig_masks[i],
        )
        alpha = unnaturalness_to_alpha(score, low=thresh_low,
                                      high=thresh_high, max_alpha=max_alpha)
        all_alphas.append(alpha)
        mouth_region = score[orig_masks[i] > 0.5]
        all_scores.append({
            "mean_score": float(mouth_region.mean()) if len(mouth_region) else 0,
            "max_score": float(mouth_region.max()) if len(mouth_region) else 0,
        })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_frames}", end="\r")

    # Temporal smoothing on alpha (per-pixel Gaussian along time axis)
    print(f"\n  Phase 3: Temporal smoothing + blending...")
    alpha_stack = np.array(all_alphas)  # (N, H, W)

    # 1D Gaussian kernel for temporal smoothing
    half_win = temporal_window // 2
    kernel_t = cv2.getGaussianKernel(temporal_window, 0).flatten()
    kernel_t = kernel_t / kernel_t.sum()

    # Pad and convolve along time axis
    smoothed_alphas = np.zeros_like(alpha_stack)
    for i in range(n_frames):
        start = max(0, i - half_win)
        end = min(n_frames, i + half_win + 1)
        weights = kernel_t[max(0, half_win - i):half_win + min(n_frames - i, half_win + 1)]
        weights = weights / weights.sum()
        smoothed_alphas[i] = np.average(
            alpha_stack[start:end], axis=0, weights=weights
        )

    # Spatial smoothing
    for i in range(n_frames):
        smoothed_alphas[i] = cv2.GaussianBlur(smoothed_alphas[i], (11, 11), 0)

    # Blend + encode via ffmpeg pipe (avoids moov atom issues with cv2)
    output_file = OUT_DIR / f"test_9_{output_suffix}.mp4"
    pipe_cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(synced_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart", "-shortest",
        str(output_file),
    ]
    proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    alpha_means = []
    alpha_maxes = []
    for i in range(n_frames):
        alpha3 = smoothed_alphas[i][..., np.newaxis]
        orig_f = orig_frames[i].astype(np.float32)
        sync_f = sync_frames[i].astype(np.float32)
        blended = (sync_f * alpha3 + orig_f * (1.0 - alpha3)).astype(np.uint8)
        proc.stdin.write(blended.tobytes())
        alpha_means.append(float(smoothed_alphas[i].mean()))
        alpha_maxes.append(float(smoothed_alphas[i].max()))

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_frames}", end="\r")

    proc.stdin.close()
    proc.wait()
    Path(tmp_clip).unlink(missing_ok=True)

    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / (1024 * 1024)
    avg_mean_alpha = float(np.mean(alpha_means))
    avg_max_alpha = float(np.mean(alpha_maxes))

    result = {
        "test_id": 9,
        "name": f"{output_suffix}",
        "person": name,
        "output_file": str(output_file),
        "total_frames": n_frames,
        "avg_mean_alpha": round(avg_mean_alpha, 4),
        "avg_max_alpha": round(avg_max_alpha, 4),
        "size_mb": round(size_mb, 1),
        "runtime_seconds": round(elapsed, 1),
        "temporal_window": temporal_window,
        "max_alpha": max_alpha,
        "score_samples": all_scores[:10],
    }

    stats_file = OUT_DIR / f"test_9_{output_suffix}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Done: {output_file.name} ({size_mb:.1f} MB, {elapsed:.0f}s)")
    print(f"  avg_mean_alpha={avg_mean_alpha:.4f}, avg_max_alpha={avg_max_alpha:.4f}")
    print(f"  avg_unnaturalness={np.mean([s['mean_score'] for s in all_scores]):.3f}")

    # Free memory
    del orig_frames, sync_frames, alpha_stack, smoothed_alphas
    return result


def main():
    p = argparse.ArgumentParser(
        description="Temporal stats judge for lip-sync blending"
    )
    p.add_argument("--only", type=str, default="Surya",
                   help="Process only this person (default: Surya)")
    p.add_argument("--all", action="store_true",
                   help="Process all lip-synced members")
    p.add_argument("--temporal-window", type=int, default=7,
                   help="Temporal smoothing window (default: 7)")
    p.add_argument("--max-alpha", type=float, default=0.8,
                   help="Max blend alpha (default: 0.8)")
    p.add_argument("--thresh-low", type=float, default=0.3,
                   help="Unnaturalness score below this = natural (default: 0.3)")
    p.add_argument("--thresh-high", type=float, default=1.0,
                   help="Unnaturalness score above this = artifact (default: 1.0)")
    args = p.parse_args()

    entries = load_analysis()

    for entry in entries:
        if entry.get("skip_lipsync"):
            continue
        name = entry["name"]
        if not args.all and name.lower() != args.only.lower():
            continue

        print(f"\n{'='*60}")
        print(f"  Temporal Judge: {name}")
        print(f"  thresholds: low={args.thresh_low}, high={args.thresh_high}")
        print(f"  max_alpha={args.max_alpha}, temporal_window={args.temporal_window}")
        print(f"{'='*60}")

        suffix = f"judge_{name.lower()}"
        process_person(
            entry, output_suffix=suffix,
            temporal_window=args.temporal_window,
            max_alpha=args.max_alpha,
            thresh_low=args.thresh_low,
            thresh_high=args.thresh_high,
        )


if __name__ == "__main__":
    main()
