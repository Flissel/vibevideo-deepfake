"""
deep_analysis.py - Deep per-person lip sync quality analysis.

For each lip-synced person, compares original vs synced mouth region across:
  1. Pixel-level: mean diff, max diff, std diff, % changed pixels
  2. Structural: SSIM in mouth region
  3. Temporal: frame-to-frame jitter (motion smoothness)
  4. Frequency: high-freq energy ratio (texture preservation)
  5. Color: LAB color shift in mouth
  6. Boundary: seam visibility at mask edge

Outputs per-person JSON + summary comparison.
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path(__file__).parent.parent.parent
ANALYSIS = VIDEO_DIR / "analysis.json"
OUT_DIR = VIDEO_DIR / "deep_analysis"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_analysis():
    with open(ANALYSIS, encoding="utf-8") as f:
        return json.load(f)


def extract_clip(video_path, clip_start, clip_end, out_path):
    ffmpeg = get_ffmpeg()
    subprocess.run([
        ffmpeg, "-y", "-i", str(video_path),
        "-ss", str(clip_start), "-t", str(clip_end - clip_start),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        str(out_path),
    ], capture_output=True, check=True)


def read_all_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def get_mouth_mask(frame_rgb, landmarker):
    """Get mouth region mask using MediaPipe."""
    import mediapipe as mp
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    )
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    h, w = frame_rgb.shape[:2]

    # Mouth landmarks (MediaPipe indices)
    MOUTH_IDX = list(range(61, 69)) + list(range(78, 96)) + [0, 13, 14, 17,
        37, 39, 40, 185, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17,
        84, 181, 91, 146]
    pts = []
    for idx in MOUTH_IDX:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])
    if len(pts) < 6:
        return None

    pts = np.array(pts, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    # Expand slightly
    face_height = max(1, int(h * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (face_height * 2, face_height * 2))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def get_boundary_mask(mouth_mask, width=5):
    """Get a thin band at the boundary of the mouth mask."""
    if mouth_mask is None:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (width * 2 + 1, width * 2 + 1))
    dilated = cv2.dilate(mouth_mask, kernel, iterations=1)
    eroded = cv2.erode(mouth_mask, kernel, iterations=1)
    boundary = dilated.astype(np.int16) - eroded.astype(np.int16)
    boundary = np.clip(boundary, 0, 255).astype(np.uint8)
    return boundary


def compute_ssim_region(img1, img2, mask):
    """Compute SSIM only in masked region."""
    from skimage.metrics import structural_similarity
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Crop to mask bounding box for efficiency
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 1.0
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    crop1 = gray1[y1:y2, x1:x2]
    crop2 = gray2[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    if crop1.shape[0] < 7 or crop1.shape[1] < 7:
        return 1.0

    ssim_map = structural_similarity(crop1, crop2, full=True)[1]
    masked_ssim = ssim_map[crop_mask > 0]
    return float(masked_ssim.mean()) if len(masked_ssim) > 0 else 1.0


def compute_hf_energy(img, mask):
    """Compute high-frequency energy ratio in masked region.
    Higher = more texture detail preserved."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    hf = gray - blurred  # high-freq component

    region = mask > 0
    if not region.any():
        return 0.0
    hf_energy = float(np.mean(hf[region] ** 2))
    total_energy = float(np.mean(gray[region] ** 2)) + 1e-6
    return hf_energy / total_energy


def compute_lab_shift(img1, img2, mask):
    """Compute mean LAB color shift in masked region."""
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    region = mask > 0
    if not region.any():
        return {"L": 0, "a": 0, "b": 0}
    diff = lab2[region] - lab1[region]
    return {
        "L": float(diff[:, 0].mean()),
        "a": float(diff[:, 1].mean()),
        "b": float(diff[:, 2].mean()),
    }


def analyze_person(entry):
    """Full deep analysis for one person."""
    name = entry["name"]
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    clip_start = entry.get("clip_start", 0)
    clip_end = entry.get("clip_end", entry["duration"])
    synced_path = VIDEO_DIR / "lip_sync" / f"{name}.mp4"

    if not synced_path.exists():
        print(f"  SKIP: {synced_path} not found")
        return None

    # Extract original clip
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_clip = tmp.name
    extract_clip(VIDEO_DIR / entry["file"], clip_start, clip_end, tmp_clip)

    print(f"  Reading frames...")
    orig_frames, fps = read_all_frames(tmp_clip)
    sync_frames, _ = read_all_frames(synced_path)
    n_frames = min(len(orig_frames), len(sync_frames))
    orig_frames = orig_frames[:n_frames]
    sync_frames = sync_frames[:n_frames]

    # Resize if needed
    h, w = sync_frames[0].shape[:2]
    for i in range(n_frames):
        if orig_frames[i].shape[:2] != (h, w):
            orig_frames[i] = cv2.resize(orig_frames[i], (w, h))

    print(f"  Creating face landmarker...")
    import mediapipe as mp
    model_path = str(VIDEO_DIR / "face_landmarker.task")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # Per-frame metrics
    frame_metrics = []
    prev_sync = None
    prev_orig = None

    # Accumulators for summary
    all_ssim = []
    all_mean_diff = []
    all_max_diff = []
    all_pct_changed = []
    all_hf_orig = []
    all_hf_sync = []
    all_jitter_sync = []
    all_jitter_orig = []
    all_boundary_diff = []
    all_lab_L = []
    all_lab_a = []
    all_lab_b = []

    print(f"  Analyzing {n_frames} frames...")
    for i in range(n_frames):
        orig = orig_frames[i]
        sync = sync_frames[i]
        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        mouth_mask = get_mouth_mask(rgb, landmarker)
        if mouth_mask is None:
            continue

        region = mouth_mask > 0
        n_pixels = int(region.sum())

        # 1. Pixel diff
        diff = np.abs(orig.astype(np.float32) - sync.astype(np.float32))
        mouth_diff = diff[region]
        mean_diff = float(mouth_diff.mean())
        max_diff = float(mouth_diff.max())
        std_diff = float(mouth_diff.std())
        pct_changed = float((mouth_diff.mean(axis=1) > 10).sum() / max(n_pixels, 1) * 100)

        # 2. SSIM
        ssim_val = compute_ssim_region(orig, sync, mouth_mask)

        # 3. Temporal jitter (frame-to-frame diff in synced)
        jitter_sync = 0.0
        jitter_orig = 0.0
        if prev_sync is not None:
            sync_temporal = np.abs(sync.astype(np.float32) - prev_sync.astype(np.float32))
            jitter_sync = float(sync_temporal[region].mean())
            orig_temporal = np.abs(orig.astype(np.float32) - prev_orig.astype(np.float32))
            jitter_orig = float(orig_temporal[region].mean())

        # 4. High-freq energy
        hf_orig = compute_hf_energy(orig, mouth_mask)
        hf_sync = compute_hf_energy(sync, mouth_mask)

        # 5. LAB shift
        lab_shift = compute_lab_shift(orig, sync, mouth_mask)

        # 6. Boundary diff
        boundary_mask = get_boundary_mask(mouth_mask, width=4)
        boundary_diff = 0.0
        if boundary_mask is not None:
            b_region = boundary_mask > 0
            if b_region.any():
                b_diff = np.abs(orig.astype(np.float32) - sync.astype(np.float32))
                boundary_diff = float(b_diff[b_region].mean())

        fm = {
            "frame": i,
            "time": round(i / fps, 3),
            "mean_diff": round(mean_diff, 2),
            "max_diff": round(max_diff, 2),
            "std_diff": round(std_diff, 2),
            "pct_changed": round(pct_changed, 1),
            "ssim": round(ssim_val, 4),
            "jitter_sync": round(jitter_sync, 2),
            "jitter_orig": round(jitter_orig, 2),
            "hf_orig": round(hf_orig, 6),
            "hf_sync": round(hf_sync, 6),
            "lab_L": round(lab_shift["L"], 2),
            "lab_a": round(lab_shift["a"], 2),
            "lab_b": round(lab_shift["b"], 2),
            "boundary_diff": round(boundary_diff, 2),
            "n_mouth_pixels": n_pixels,
        }
        frame_metrics.append(fm)

        all_ssim.append(ssim_val)
        all_mean_diff.append(mean_diff)
        all_max_diff.append(max_diff)
        all_pct_changed.append(pct_changed)
        all_hf_orig.append(hf_orig)
        all_hf_sync.append(hf_sync)
        all_jitter_sync.append(jitter_sync)
        all_jitter_orig.append(jitter_orig)
        all_boundary_diff.append(boundary_diff)
        all_lab_L.append(lab_shift["L"])
        all_lab_a.append(lab_shift["a"])
        all_lab_b.append(lab_shift["b"])

        prev_sync = sync
        prev_orig = orig

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_frames}")

    landmarker.close()
    Path(tmp_clip).unlink(missing_ok=True)

    # Jitter ratio: >1 means synced is jumpier than original
    jitter_orig_arr = np.array(all_jitter_orig[1:])  # skip first (no prev)
    jitter_sync_arr = np.array(all_jitter_sync[1:])
    jitter_ratio = float(jitter_sync_arr.mean() / max(jitter_orig_arr.mean(), 0.01))

    # HF preservation: ratio of synced/original high-freq energy
    hf_ratio = float(np.mean(all_hf_sync) / max(np.mean(all_hf_orig), 1e-8))

    summary = {
        "name": name,
        "total_frames": n_frames,
        "analyzed_frames": len(frame_metrics),
        "fps": fps,
        "avg_ssim": round(float(np.mean(all_ssim)), 4),
        "min_ssim": round(float(np.min(all_ssim)), 4),
        "avg_mean_diff": round(float(np.mean(all_mean_diff)), 2),
        "max_of_max_diff": round(float(np.max(all_max_diff)), 2),
        "avg_pct_changed": round(float(np.mean(all_pct_changed)), 1),
        "avg_jitter_sync": round(float(np.mean(all_jitter_sync[1:])), 2),
        "avg_jitter_orig": round(float(np.mean(all_jitter_orig[1:])), 2),
        "jitter_ratio": round(jitter_ratio, 3),
        "hf_energy_orig": round(float(np.mean(all_hf_orig)), 6),
        "hf_energy_sync": round(float(np.mean(all_hf_sync)), 6),
        "hf_preservation": round(hf_ratio, 4),
        "avg_boundary_diff": round(float(np.mean(all_boundary_diff)), 2),
        "avg_lab_L_shift": round(float(np.mean(all_lab_L)), 2),
        "avg_lab_a_shift": round(float(np.mean(all_lab_a)), 2),
        "avg_lab_b_shift": round(float(np.mean(all_lab_b)), 2),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    # Quality grades
    grades = {}
    grades["ssim"] = "A" if summary["avg_ssim"] > 0.85 else "B" if summary["avg_ssim"] > 0.75 else "C" if summary["avg_ssim"] > 0.65 else "D"
    grades["jitter"] = "A" if jitter_ratio < 1.2 else "B" if jitter_ratio < 1.5 else "C" if jitter_ratio < 2.0 else "D"
    grades["texture"] = "A" if hf_ratio > 0.8 else "B" if hf_ratio > 0.6 else "C" if hf_ratio > 0.4 else "D"
    grades["boundary"] = "A" if summary["avg_boundary_diff"] < 8 else "B" if summary["avg_boundary_diff"] < 15 else "C" if summary["avg_boundary_diff"] < 25 else "D"
    grades["color"] = "A" if abs(summary["avg_lab_a_shift"]) < 2 else "B" if abs(summary["avg_lab_a_shift"]) < 5 else "C"
    summary["grades"] = grades

    # Overall grade
    grade_vals = {"A": 4, "B": 3, "C": 2, "D": 1}
    avg_grade = np.mean([grade_vals[g] for g in grades.values()])
    summary["overall_grade"] = "A" if avg_grade >= 3.5 else "B" if avg_grade >= 2.5 else "C" if avg_grade >= 1.5 else "D"

    result = {"summary": summary, "frames": frame_metrics}

    out_file = OUT_DIR / f"{name}_deep.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  === {name} ===")
    print(f"  SSIM:        {summary['avg_ssim']:.4f} (grade: {grades['ssim']})")
    print(f"  Jitter:      sync={summary['avg_jitter_sync']:.1f} vs orig={summary['avg_jitter_orig']:.1f} ratio={jitter_ratio:.2f} (grade: {grades['jitter']})")
    print(f"  Texture(HF): {hf_ratio:.3f} preservation (grade: {grades['texture']})")
    print(f"  Boundary:    {summary['avg_boundary_diff']:.1f} avg diff (grade: {grades['boundary']})")
    print(f"  Color shift: L={summary['avg_lab_L_shift']:.1f} a={summary['avg_lab_a_shift']:.1f} b={summary['avg_lab_b_shift']:.1f} (grade: {grades['color']})")
    print(f"  Overall:     {summary['overall_grade']}")
    print(f"  Saved: {out_file.name}")

    return summary


def main():
    entries = load_analysis()
    OUT_DIR.mkdir(exist_ok=True)

    summaries = []
    for entry in entries:
        if entry.get("skip_lipsync"):
            continue
        name = entry["name"]
        synced = VIDEO_DIR / "lip_sync" / f"{name}.mp4"
        if not synced.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  Deep Analysis: {name}")
        print(f"{'='*60}")

        s = analyze_person(entry)
        if s:
            summaries.append(s)

    # Print comparison table
    if summaries:
        print(f"\n\n{'='*80}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"  {'Name':<12} {'SSIM':>6} {'Jitter':>8} {'HF':>6} {'Bound':>7} {'Color':>7} {'GRADE':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
        for s in summaries:
            g = s["grades"]
            print(f"  {s['name']:<12} {s['avg_ssim']:.3f}  {s['jitter_ratio']:>6.2f}x  {s['hf_preservation']:.3f}  {s['avg_boundary_diff']:>5.1f}  "
                  f"a={s['avg_lab_a_shift']:>4.1f}  {s['overall_grade']:>5}")

        # Save comparison
        comp_file = OUT_DIR / "comparison.json"
        with open(comp_file, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\n  Saved: {comp_file}")


if __name__ == "__main__":
    main()
