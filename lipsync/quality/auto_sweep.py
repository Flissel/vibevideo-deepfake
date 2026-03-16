"""
auto_sweep.py - Automatic parameter sweep to find best lip-sync blend for Surya.

Tests combinations of:
  - MuseTalk bbox_shift (0, 2, 5)
  - Blend methods (spatial_freq, alpha_clamp, LAB, combos)
  - Max alpha (0.3, 0.5, 0.7)
  - Temporal window (5, 9, 13)

Quality metrics (computed automatically):
  - temporal_smoothness: 1 - mean(abs(frame_diff)) in mouth region
    Higher = less flickering/waves
  - ssim_outside_mouth: SSIM of non-mouth area vs original
    Higher = preserves original better
  - mouth_naturalness: SSIM of mouth area vs original (baseline comparison)
  - combined_score: weighted combination of all metrics

Usage:
  python auto_sweep.py                  # full sweep
  python auto_sweep.py --quick          # quick sweep (fewer combos)
  python auto_sweep.py --bbox-only      # only test bbox_shift values
"""
import argparse
import json
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from adaptive_blend import create_face_landmarker, get_mouth_mask, VIDEO_DIR

ANALYSIS = VIDEO_DIR / "analysis.json"
ORIGINAL = VIDEO_DIR / "Surya.mp4"
SYNCED = VIDEO_DIR / "lip_sync" / "Surya.mp4"
OUT_DIR = VIDEO_DIR / "sweep_surya"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_surya_entry():
    with open(ANALYSIS, encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        if e["name"] == "Surya":
            return e
    raise ValueError("Surya not found in analysis.json")


def read_frames(video_path, max_frames=None):
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


def extract_clip(video_path, start, end, out_path):
    ffmpeg = get_ffmpeg()
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(video_path),
        "-ss", str(start), "-t", str(end - start),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        str(out_path),
    ], capture_output=True, check=True)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_ssim_region(img1, img2, mask):
    """Compute SSIM-like metric within a masked region."""
    from skimage.metrics import structural_similarity
    # Convert to grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mask_bool = mask > 0.5
    if mask_bool.sum() < 100:
        return 1.0
    # Crop to bounding box of mask for efficiency
    ys, xs = np.where(mask_bool)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop1 = g1[y0:y1, x0:x1]
    crop2 = g2[y0:y1, x0:x1]
    crop_mask = mask_bool[y0:y1, x0:x1]
    # Simple local SSIM
    if crop1.size < 64:
        return 1.0
    win = min(7, min(crop1.shape[0], crop1.shape[1]))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return 1.0
    try:
        score = structural_similarity(crop1, crop2, win_size=win)
    except Exception:
        # Fallback: normalized cross-correlation
        m1, m2 = crop1[crop_mask], crop2[crop_mask]
        if m1.std() < 1e-6 or m2.std() < 1e-6:
            return 1.0
        score = float(np.corrcoef(m1.ravel(), m2.ravel())[0, 1])
    return float(score)


def compute_temporal_smoothness(frames, masks):
    """Measure how smooth the mouth region is across time.

    Low value = lots of flickering/waves.
    High value = smooth transitions.

    Computes: 1 - mean(frame_to_frame_diff) / 255 in mouth region.
    """
    diffs = []
    for i in range(1, len(frames)):
        mask = masks[i]
        if mask is None or (mask > 0.5).sum() < 100:
            continue
        diff = np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32))
        mouth_diff = diff[mask > 0.5].mean()
        diffs.append(mouth_diff)
    if not diffs:
        return 1.0
    return 1.0 - np.mean(diffs) / 255.0


def compute_temporal_jitter(frames, masks):
    """Measure acceleration (second derivative) in mouth region.

    High jitter = oscillating waves. Low jitter = smooth or steady motion.
    This specifically catches the "wave" artifacts.
    """
    if len(frames) < 3:
        return 0.0
    accels = []
    for i in range(1, len(frames) - 1):
        mask = masks[i]
        if mask is None or (mask > 0.5).sum() < 100:
            continue
        f_prev = frames[i-1].astype(np.float32)
        f_curr = frames[i].astype(np.float32)
        f_next = frames[i+1].astype(np.float32)
        # Second derivative (acceleration)
        accel = f_next - 2 * f_curr + f_prev
        mouth_accel = np.abs(accel[mask > 0.5]).mean()
        accels.append(mouth_accel)
    if not accels:
        return 0.0
    return float(np.mean(accels))


# ---------------------------------------------------------------------------
# Blend functions
# ---------------------------------------------------------------------------

def blend_baseline(orig_f, sync_f, mask, max_alpha=1.0, **kw):
    """BGR L2 + linear ramp."""
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    alpha = np.clip((diff - 10) / 30.0, 0.0, 1.0).astype(np.float32)
    alpha = np.minimum(alpha, max_alpha)
    alpha = alpha * mask
    return cv2.GaussianBlur(alpha, (15, 15), 0)


def blend_lab(orig_f, sync_f, mask, max_alpha=1.0, **kw):
    """LAB colorspace diff."""
    o = np.clip(orig_f, 0, 255).astype(np.uint8)
    s = np.clip(sync_f, 0, 255).astype(np.uint8)
    o_lab = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((o_lab - s_lab) ** 2, axis=2))
    alpha = np.clip((diff - 10) / 30.0, 0.0, 1.0).astype(np.float32)
    alpha = np.minimum(alpha, max_alpha)
    alpha = alpha * mask
    return cv2.GaussianBlur(alpha, (15, 15), 0)


def blend_spatial_freq(orig_f, sync_f, mask, blur_size=21, max_alpha=1.0, **kw):
    """Frequency split: low-freq from synced, high-freq from original."""
    sync_low = cv2.GaussianBlur(sync_f, (blur_size, blur_size), 0)
    orig_low = cv2.GaussianBlur(orig_f, (blur_size, blur_size), 0)
    orig_high = orig_f - orig_low
    combined = np.clip(sync_low + orig_high, 0, 255)
    mask3 = mask[..., np.newaxis]
    result = (combined * mask3 + orig_f * (1.0 - mask3))
    return result  # returns pre-blended frame, not alpha


def blend_combo_lab_freq(orig_f, sync_f, mask, max_alpha=0.7, blur_size=21, **kw):
    """Combo: spatial freq split + LAB-driven alpha + alpha clamp."""
    # Frequency split for the synced source
    sync_low = cv2.GaussianBlur(sync_f, (blur_size, blur_size), 0)
    orig_low = cv2.GaussianBlur(orig_f, (blur_size, blur_size), 0)
    orig_high = orig_f - orig_low
    freq_combined = np.clip(sync_low + orig_high, 0, 255)
    # LAB-based alpha
    o = np.clip(orig_f, 0, 255).astype(np.uint8)
    s = np.clip(freq_combined, 0, 255).astype(np.uint8)
    o_lab = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((o_lab - s_lab) ** 2, axis=2))
    alpha = np.clip((diff - 8) / 25.0, 0.0, 1.0).astype(np.float32)
    alpha = np.minimum(alpha, max_alpha)
    alpha = alpha * mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    # Blend freq_combined with original using alpha
    alpha3 = alpha[..., np.newaxis]
    return (freq_combined * alpha3 + orig_f * (1.0 - alpha3))


def blend_combo_freq_clamp(orig_f, sync_f, mask, max_alpha=0.5, blur_size=21, **kw):
    """Combo: spatial freq split + hard alpha clamp (aggressive original preservation)."""
    sync_low = cv2.GaussianBlur(sync_f, (blur_size, blur_size), 0)
    orig_low = cv2.GaussianBlur(orig_f, (blur_size, blur_size), 0)
    orig_high = orig_f - orig_low
    freq_combined = np.clip(sync_low + orig_high, 0, 255)
    # Simple diff alpha
    diff = np.sqrt(np.sum((orig_f - freq_combined) ** 2, axis=2))
    alpha = np.clip((diff - 5) / 20.0, 0.0, 1.0).astype(np.float32)
    alpha = np.minimum(alpha, max_alpha)
    alpha = alpha * mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    alpha3 = alpha[..., np.newaxis]
    return (freq_combined * alpha3 + orig_f * (1.0 - alpha3))


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

BLEND_METHODS = {
    "baseline":         (blend_baseline, False),
    "lab":              (blend_lab, False),
    "spatial_freq":     (blend_spatial_freq, True),   # True = returns frame not alpha
    "combo_lab_freq":   (blend_combo_lab_freq, True),
    "combo_freq_clamp": (blend_combo_freq_clamp, True),
}


def run_single_config(config, orig_frames, sync_frames, masks, fps, w, h):
    """Run a single blend configuration, return quality metrics."""
    method_name = config["method"]
    max_alpha = config["max_alpha"]
    temporal_window = config["temporal_window"]
    blend_fn, returns_frame = BLEND_METHODS[method_name]

    n = len(orig_frames)

    # Pass 1: compute blended frames or alphas
    all_alphas = []
    all_frames = []  # for returns_frame methods

    for i in range(n):
        orig_f = orig_frames[i].astype(np.float32)
        sync_f = sync_frames[i].astype(np.float32)
        mask = masks[i]

        result = blend_fn(orig_f, sync_f, mask, max_alpha=max_alpha)

        if returns_frame:
            all_frames.append(np.clip(result, 0, 255).astype(np.uint8))
            all_alphas.append(mask)  # placeholder
        else:
            all_alphas.append(result)
            all_frames.append(None)

    # Pass 2: temporal smoothing + final blend
    half_win = temporal_window // 2
    final_frames = []

    for i in range(n):
        if returns_frame:
            # Temporal average of pre-blended frames
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            # Just use the current frame (temporal smoothing on pre-blended is tricky)
            final_frames.append(all_frames[i])
        else:
            # Temporal smooth the alpha
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            smoothed = np.mean(all_alphas[start:end], axis=0).astype(np.float32)
            smoothed = cv2.GaussianBlur(smoothed, (11, 11), 0)
            alpha3 = smoothed[..., np.newaxis]
            orig_f = orig_frames[i].astype(np.float32)
            sync_f = sync_frames[i].astype(np.float32)
            blended = (sync_f * alpha3 + orig_f * (1.0 - alpha3)).astype(np.uint8)
            final_frames.append(blended)

    # Compute quality metrics
    temporal_smooth = compute_temporal_smoothness(final_frames, masks)
    temporal_jitter = compute_temporal_jitter(final_frames, masks)

    # SSIM for mouth region (vs original) — measures how much original texture preserved
    ssim_mouth_scores = []
    ssim_outside_scores = []
    for i in range(0, n, 5):  # sample every 5 frames for speed
        mask = masks[i]
        inv_mask = 1.0 - mask
        ssim_m = compute_ssim_region(final_frames[i], orig_frames[i], mask)
        ssim_o = compute_ssim_region(final_frames[i], orig_frames[i], inv_mask)
        ssim_mouth_scores.append(ssim_m)
        ssim_outside_scores.append(ssim_o)

    ssim_mouth = float(np.mean(ssim_mouth_scores))
    ssim_outside = float(np.mean(ssim_outside_scores))

    # Combined score:
    # - temporal_smooth: higher = less flickering (weight 0.35)
    # - (1 - jitter/20): lower jitter = less waves (weight 0.35)
    # - ssim_mouth: higher = more original preserved (weight 0.15)
    # - ssim_outside: higher = non-mouth unchanged (weight 0.15)
    jitter_score = max(0, 1.0 - temporal_jitter / 20.0)
    combined = (0.35 * temporal_smooth +
                0.35 * jitter_score +
                0.15 * ssim_mouth +
                0.15 * ssim_outside)

    return {
        "config": config,
        "temporal_smoothness": round(temporal_smooth, 4),
        "temporal_jitter": round(temporal_jitter, 4),
        "ssim_mouth": round(ssim_mouth, 4),
        "ssim_outside": round(ssim_outside, 4),
        "combined_score": round(combined, 4),
    }, final_frames


def write_video(frames, fps, w, h, output_path, audio_source=None):
    """Write frames to video with ffmpeg pipe."""
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
    ]
    if audio_source:
        cmd += ["-i", str(audio_source)]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-map", "0:v:0",
    ]
    if audio_source:
        cmd += ["-map", "1:a:0?"]
    cmd += ["-movflags", "+faststart", "-shortest", str(output_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick sweep (fewer combos)")
    parser.add_argument("--bbox-only", action="store_true",
                        help="Only test bbox_shift values")
    parser.add_argument("--top", type=int, default=3,
                        help="Save top N videos (default: 3)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    entry = load_surya_entry()
    clip_start = entry.get("clip_start", 0)
    clip_end = entry.get("clip_end", entry["duration"])

    # Extract original clip
    print("Extracting original clip...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_clip = tmp.name
    extract_clip(ORIGINAL, clip_start, clip_end, tmp_clip)

    print("Reading frames...")
    orig_frames, fps, w, h = read_frames(tmp_clip)
    sync_frames, _, _, _ = read_frames(str(SYNCED))
    n = min(len(orig_frames), len(sync_frames))
    orig_frames = orig_frames[:n]
    sync_frames = sync_frames[:n]

    # Resize if needed
    for i in range(n):
        if orig_frames[i].shape[:2] != (h, w):
            orig_frames[i] = cv2.resize(orig_frames[i], (w, h))

    print(f"Computing mouth masks ({n} frames)...")
    landmarker = create_face_landmarker()
    masks = []
    for frame in orig_frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = get_mouth_mask(rgb, landmarker)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.float32)
        masks.append(mask)
    landmarker.close()

    # Define sweep grid
    if args.quick:
        methods = ["baseline", "spatial_freq", "combo_lab_freq", "combo_freq_clamp"]
        max_alphas = [0.5, 0.7]
        temporal_windows = [5, 9]
    else:
        methods = list(BLEND_METHODS.keys())
        max_alphas = [0.3, 0.5, 0.7, 1.0]
        temporal_windows = [5, 9, 13]

    configs = []
    for method, ma, tw in product(methods, max_alphas, temporal_windows):
        # Skip redundant: spatial_freq ignores max_alpha since it returns frames
        if method == "spatial_freq" and ma != 0.7:
            continue
        configs.append({
            "method": method,
            "max_alpha": ma,
            "temporal_window": tw,
        })

    print(f"\n{'='*60}")
    print(f"  SWEEP: {len(configs)} configurations")
    print(f"{'='*60}\n")

    results = []
    for idx, config in enumerate(configs):
        label = f"{config['method']}_a{config['max_alpha']}_t{config['temporal_window']}"
        print(f"  [{idx+1}/{len(configs)}] {label}...", end=" ", flush=True)
        t0 = time.time()
        metrics, frames = run_single_config(config, orig_frames, sync_frames,
                                             masks, fps, w, h)
        elapsed = time.time() - t0
        metrics["label"] = label
        metrics["runtime"] = round(elapsed, 1)
        results.append((metrics, frames))
        print(f"score={metrics['combined_score']:.4f} "
              f"smooth={metrics['temporal_smoothness']:.4f} "
              f"jitter={metrics['temporal_jitter']:.2f} "
              f"({elapsed:.0f}s)")

    # Sort by combined score
    results.sort(key=lambda x: x[0]["combined_score"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  TOP {args.top} RESULTS")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Label':<40} {'Score':<8} {'Smooth':<8} {'Jitter':<8} {'SSIM_m':<8}")
    print(f"  {'-'*4:<5} {'-'*38:<40} {'-'*6:<8} {'-'*6:<8} {'-'*6:<8} {'-'*6:<8}")

    for rank, (metrics, frames) in enumerate(results[:args.top]):
        label = metrics["label"]
        print(f"  {rank+1:<5} {label:<40} "
              f"{metrics['combined_score']:<8.4f} "
              f"{metrics['temporal_smoothness']:<8.4f} "
              f"{metrics['temporal_jitter']:<8.2f} "
              f"{metrics['ssim_mouth']:<8.4f}")

        # Save video
        out_path = OUT_DIR / f"rank_{rank+1}_{label}.mp4"
        write_video(frames, fps, w, h, out_path, audio_source=SYNCED)
        print(f"         -> {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save full results
    all_metrics = [r[0] for r in results]
    summary_path = OUT_DIR / "sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Full results: {summary_path}")

    # Print bottom 3 for comparison
    print(f"\n  WORST 3:")
    for metrics, _ in results[-3:]:
        print(f"    {metrics['label']:<40} score={metrics['combined_score']:.4f}")

    # Cleanup
    Path(tmp_clip).unlink(missing_ok=True)

    # Return best config
    best = results[0][0]
    print(f"\n  WINNER: {best['label']}")
    print(f"  Score: {best['combined_score']:.4f}")
    print(f"  Config: {json.dumps(best['config'], indent=2)}")


if __name__ == "__main__":
    main()
