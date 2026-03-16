"""
test_enhancements.py - Test 8 adaptive blend enhancements on Surya.

Each test produces a separate high-quality video in test_surya/.
Run with: python test_enhancements.py --all
"""
import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from adaptive_blend import (
    create_face_landmarker,
    get_mouth_mask,
    MOUTH_OUTER,
    MOUTH_INNER,
    VIDEO_DIR,
)

ORIGINAL = VIDEO_DIR / "Surya.mp4"
SYNCED = VIDEO_DIR / "lip_sync" / "Surya.mp4"
OUT_DIR = VIDEO_DIR / "test_surya"
CLIP_START, CLIP_END = 0, 10
THRESH_LOW, THRESH_HIGH = 10, 40
TEMPORAL_WINDOW = 5


# ---------------------------------------------------------------------------
# 8 blend functions — all share signature:
#   (orig_f, sync_f, mouth_mask, frame_idx, context) -> (alpha, stats)
#   orig_f, sync_f: float32 BGR frames
#   mouth_mask: float32 (H,W) 0-1
#   context: mutable dict persisting across frames
# ---------------------------------------------------------------------------

def blend_baseline(orig_f, sync_f, mouth_mask, frame_idx, context):
    """Baseline: BGR L2 + linear ramp (current default)."""
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    alpha = np.clip(
        (diff - THRESH_LOW) / max(THRESH_HIGH - THRESH_LOW, 1), 0.0, 1.0
    ).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"diff_mean": float(m.mean()) if len(m) else 0}


# Test 1: LAB color space
def blend_lab_diff(orig_f, sync_f, mouth_mask, frame_idx, context):
    orig_u8 = np.clip(orig_f, 0, 255).astype(np.uint8)
    sync_u8 = np.clip(sync_f, 0, 255).astype(np.uint8)
    orig_lab = cv2.cvtColor(orig_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    sync_lab = cv2.cvtColor(sync_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((orig_lab - sync_lab) ** 2, axis=2))
    alpha = np.clip(
        (diff - THRESH_LOW) / max(THRESH_HIGH - THRESH_LOW, 1), 0.0, 1.0
    ).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"diff_mean": float(m.mean()) if len(m) else 0}


# Test 2: Per-channel weighted LAB (a* focus)
def blend_lab_weighted(orig_f, sync_f, mouth_mask, frame_idx, context):
    weights = np.array([0.3, 0.5, 0.2], dtype=np.float32)  # L, a*, b*
    orig_u8 = np.clip(orig_f, 0, 255).astype(np.uint8)
    sync_u8 = np.clip(sync_f, 0, 255).astype(np.uint8)
    orig_lab = cv2.cvtColor(orig_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    sync_lab = cv2.cvtColor(sync_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff_sq = (orig_lab - sync_lab) ** 2
    diff = np.sqrt(np.sum(diff_sq * weights[np.newaxis, np.newaxis, :], axis=2))
    alpha = np.clip(
        (diff - THRESH_LOW) / max(THRESH_HIGH - THRESH_LOW, 1), 0.0, 1.0
    ).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"diff_mean": float(m.mean()) if len(m) else 0}


# Test 3: Sigmoid curve
def blend_sigmoid(orig_f, sync_f, mouth_mask, frame_idx, context):
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    mid = (THRESH_LOW + THRESH_HIGH) / 2.0  # 25
    k = 0.2
    # Clip to avoid overflow in exp
    exponent = np.clip(-k * (diff - mid), -50, 50)
    alpha = (1.0 / (1.0 + np.exp(exponent))).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"diff_mean": float(m.mean()) if len(m) else 0}


# Test 4: Spatial frequency preservation
def blend_spatial_freq(orig_f, sync_f, mouth_mask, frame_idx, context):
    blur_size = 21
    sync_low = cv2.GaussianBlur(sync_f, (blur_size, blur_size), 0)
    orig_low = cv2.GaussianBlur(orig_f, (blur_size, blur_size), 0)
    orig_high = orig_f - orig_low  # texture/detail from original
    combined = np.clip(sync_low + orig_high, 0, 255)
    mask3 = mouth_mask[..., np.newaxis]
    preblended = (combined * mask3 + orig_f * (1.0 - mask3)).astype(np.uint8)
    context["preblended"] = preblended
    alpha = mouth_mask.copy()
    return alpha, {"blend_type": "freq_split"}


# Test 5: Per-frame adaptive thresholds
def blend_adaptive_thresh(orig_f, sync_f, mouth_mask, frame_idx, context):
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    mouth_pixels = diff[mouth_mask > 0.5]
    if len(mouth_pixels) > 100:
        t_low = float(np.percentile(mouth_pixels, 25))
        t_high = float(np.percentile(mouth_pixels, 75))
    else:
        t_low, t_high = THRESH_LOW, THRESH_HIGH
    rng = max(t_high - t_low, 1.0)
    alpha = np.clip((diff - t_low) / rng, 0.0, 1.0).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    return alpha, {"t_low": t_low, "t_high": t_high}


# Test 6: Temporal decay (EMA) — sets flag for harness
def blend_ema(orig_f, sync_f, mouth_mask, frame_idx, context):
    context["use_ema"] = True
    context["ema_decay"] = 0.7
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    alpha = np.clip(
        (diff - THRESH_LOW) / max(THRESH_HIGH - THRESH_LOW, 1), 0.0, 1.0
    ).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"diff_mean": float(m.mean()) if len(m) else 0}


# Test 7: Larger mask feather
def get_mouth_mask_feathered(frame_rgb, landmarker, expand_ratio=0.08,
                              feather_ratio=0.12):
    """Same as get_mouth_mask but with larger feather radius."""
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)
    blur_r = max(5, int(face_height * feather_ratio))
    ksize = blur_r * 2 + 1
    mask_f = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0, (ksize, ksize), blur_r
    )
    return mask_f


def blend_feather(orig_f, sync_f, mouth_mask, frame_idx, context):
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    alpha = np.clip(
        (diff - THRESH_LOW) / max(THRESH_HIGH - THRESH_LOW, 1), 0.0, 1.0
    ).astype(np.float32)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"feather_ratio": 0.12,
                   "diff_mean": float(m.mean()) if len(m) else 0}


# Test 8: Alpha clamp
def blend_alpha_clamp(orig_f, sync_f, mouth_mask, frame_idx, context):
    diff = np.sqrt(np.sum((orig_f - sync_f) ** 2, axis=2))
    alpha = np.clip(
        (diff - THRESH_LOW) / max(THRESH_HIGH - THRESH_LOW, 1), 0.0, 1.0
    ).astype(np.float32)
    alpha = np.minimum(alpha, 0.7)
    alpha = alpha * mouth_mask
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    m = diff[mouth_mask > 0.5]
    return alpha, {"max_alpha": 0.7,
                   "diff_mean": float(m.mean()) if len(m) else 0}


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = {
    0: ("baseline",          blend_baseline,        None),
    1: ("lab_colorspace",    blend_lab_diff,         None),
    2: ("lab_weighted_a",    blend_lab_weighted,      None),
    3: ("sigmoid_curve",     blend_sigmoid,           None),
    4: ("spatial_freq",      blend_spatial_freq,      None),
    5: ("adaptive_thresh",   blend_adaptive_thresh,   None),
    6: ("temporal_ema",      blend_ema,               None),
    7: ("mask_feather",      blend_feather,           get_mouth_mask_feathered),
    8: ("alpha_clamp_07",    blend_alpha_clamp,       None),
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_test(test_id, name, blend_fn, custom_mask_fn=None):
    """Run a single test, return stats dict."""
    t0 = time.time()
    ffmpeg = get_ffmpeg()
    OUT_DIR.mkdir(exist_ok=True)

    # Extract clip 0-10s from original
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_clip = tmp.name
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(ORIGINAL),
        "-ss", str(CLIP_START), "-t", str(CLIP_END - CLIP_START),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        str(tmp_clip),
    ], capture_output=True, check=True)

    cap_orig = cv2.VideoCapture(tmp_clip)
    cap_sync = cv2.VideoCapture(str(SYNCED))

    fps = cap_sync.get(cv2.CAP_PROP_FPS)
    w = int(cap_sync.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_sync.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(
        int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_sync.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    landmarker = create_face_landmarker()
    context = {}

    # Pass 1: compute alpha maps
    all_alphas = []
    all_orig = []
    all_sync = []
    all_preblended = []
    per_frame_stats = []

    print(f"  Pass 1: computing alpha maps ({total} frames)...")
    for i in range(total):
        ret_o, frame_orig = cap_orig.read()
        ret_s, frame_sync = cap_sync.read()
        if not ret_o or not ret_s:
            break

        if frame_orig.shape[:2] != (h, w):
            frame_orig = cv2.resize(frame_orig, (w, h))

        frame_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)

        if custom_mask_fn is not None:
            mouth_mask = custom_mask_fn(frame_rgb, landmarker)
        else:
            mouth_mask = get_mouth_mask(frame_rgb, landmarker)

        if mouth_mask is None:
            mouth_mask = np.zeros((h, w), dtype=np.float32)

        orig_f = frame_orig.astype(np.float32)
        sync_f = frame_sync.astype(np.float32)

        context.pop("preblended", None)
        alpha, stats = blend_fn(orig_f, sync_f, mouth_mask, i, context)

        all_alphas.append(alpha)
        all_orig.append(frame_orig)
        all_sync.append(frame_sync)
        all_preblended.append(context.get("preblended"))
        per_frame_stats.append(stats)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total}", end="\r")

    cap_orig.release()
    cap_sync.release()
    landmarker.close()

    n_frames = len(all_alphas)
    half_win = TEMPORAL_WINDOW // 2
    use_ema = context.get("use_ema", False)
    ema_decay = context.get("ema_decay", 0.7)

    # Pass 2: temporal smoothing + blend
    tmp_video = str(OUT_DIR / f"test_{test_id}_{name}_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    alpha_stats = []
    print(f"\n  Pass 2: temporal smoothing + blending {n_frames} frames...")

    ema_state = None
    for i in range(n_frames):
        # Temporal smoothing
        if use_ema:
            if ema_state is None:
                ema_state = all_alphas[i].copy()
            else:
                ema_state = ema_decay * ema_state + (1 - ema_decay) * all_alphas[i]
            smoothed_alpha = ema_state.copy()
        else:
            start = max(0, i - half_win)
            end = min(n_frames, i + half_win + 1)
            smoothed_alpha = np.mean(all_alphas[start:end], axis=0).astype(
                np.float32
            )

        smoothed_alpha = cv2.GaussianBlur(smoothed_alpha, (9, 9), 0)

        # Blend
        if all_preblended[i] is not None:
            blended = all_preblended[i]
        else:
            alpha3 = smoothed_alpha[..., np.newaxis]
            orig_f = all_orig[i].astype(np.float32)
            sync_f = all_sync[i].astype(np.float32)
            blended = (sync_f * alpha3 + orig_f * (1.0 - alpha3)).astype(
                np.uint8
            )

        writer.write(blended)
        alpha_stats.append({
            "mean_alpha": float(smoothed_alpha.mean()),
            "max_alpha": float(smoothed_alpha.max()),
        })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_frames}", end="\r")

    writer.release()
    del all_alphas, all_orig, all_sync, all_preblended

    # Mux audio from synced video
    output_file = OUT_DIR / f"test_{test_id}_{name}.mp4"
    subprocess.run([
        ffmpeg, "-y",
        "-i", tmp_video,
        "-i", str(SYNCED),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart", "-shortest",
        str(output_file),
    ], capture_output=True, check=True)
    Path(tmp_video).unlink(missing_ok=True)
    Path(tmp_clip).unlink(missing_ok=True)

    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / (1024 * 1024)

    avg_mean_alpha = float(np.mean([s["mean_alpha"] for s in alpha_stats]))
    avg_max_alpha = float(np.mean([s["max_alpha"] for s in alpha_stats]))

    result = {
        "test_id": test_id,
        "name": name,
        "output_file": str(output_file),
        "total_frames": n_frames,
        "avg_mean_alpha": round(avg_mean_alpha, 4),
        "avg_max_alpha": round(avg_max_alpha, 4),
        "size_mb": round(size_mb, 1),
        "runtime_seconds": round(elapsed, 1),
        "per_frame_blend_stats": per_frame_stats[:5],  # sample
    }

    # Save stats
    stats_file = OUT_DIR / f"test_{test_id}_{name}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Done: {output_file.name} ({size_mb:.1f} MB, {elapsed:.0f}s)")
    print(f"  avg_mean_alpha={avg_mean_alpha:.4f}, avg_max_alpha={avg_max_alpha:.4f}")
    return result


def main():
    p = argparse.ArgumentParser(
        description="Test 8 lip-sync blending enhancements on Surya"
    )
    p.add_argument("--test", type=int, action="append",
                   help="Run specific test(s) by ID (0=baseline, 1-8)")
    p.add_argument("--all", action="store_true", help="Run all 8 tests")
    p.add_argument("--with-baseline", action="store_true",
                   help="Also run baseline (test 0) for comparison")
    args = p.parse_args()

    if args.all:
        tests_to_run = list(range(1, 9))
        if args.with_baseline:
            tests_to_run = [0] + tests_to_run
    elif args.test:
        tests_to_run = args.test
    else:
        p.error("Specify --test N or --all")

    OUT_DIR.mkdir(exist_ok=True)
    results = []

    for tid in tests_to_run:
        if tid not in TESTS:
            print(f"  Unknown test ID: {tid}")
            continue
        tname, fn, mask_fn = TESTS[tid]
        print(f"\n{'='*60}")
        print(f"  Test {tid}: {tname}")
        print(f"{'='*60}")
        result = run_test(tid, tname, fn, custom_mask_fn=mask_fn)
        results.append(result)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'ID':<4} {'Name':<20} {'mean_alpha':<12} {'max_alpha':<12} {'Size':<8} {'Time'}")
    print(f"  {'-'*4} {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['test_id']:<4} {r['name']:<20} {r['avg_mean_alpha']:<12.4f} "
              f"{r['avg_max_alpha']:<12.4f} {r['size_mb']:<8.1f} {r['runtime_seconds']:.0f}s")

    # Save combined summary
    summary_file = OUT_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
