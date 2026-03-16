"""
wave_detector.py - Detect and visualize temporal pixel waves in lip-synced video.

For each mouth pixel, extracts its brightness over time, computes FFT,
and measures energy in the "wave band" (5-15 Hz) vs "natural band" (0-3 Hz).

High wave_ratio = artifact oscillation. Low = natural movement.

Usage:
  python wave_detector.py --name Lisa
  python wave_detector.py --name Lisa --visualize
  python wave_detector.py --all
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path(__file__).parent.parent.parent
ANALYSIS = VIDEO_DIR / "analysis.json"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def create_face_landmarker():
    import mediapipe as mp
    model_path = VIDEO_DIR / "face_landmarker.task"
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


MOUTH_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
MOUTH_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]


def get_mouth_mask(frame_rgb, landmarker, expand_ratio=0.06):
    """Get mouth region mask."""
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    all_idx = list(set(MOUTH_OUTER + MOUTH_INNER))
    pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in all_idx])

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    face_height = int(abs(lm[10].y - lm[152].y) * h)
    expand_px = max(3, int(face_height * expand_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2, expand_px * 2))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask > 128


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


def analyze_waves(name, visualize=False):
    """Detect temporal waves in a person's lip-synced video."""
    with open(ANALYSIS, encoding="utf-8") as f:
        entries = json.load(f)

    entry = next((e for e in entries if e["name"] == name), None)
    if not entry:
        print(f"  {name} not found in analysis.json")
        return None

    synced_path = VIDEO_DIR / "lip_sync" / f"{name}.mp4"
    if not synced_path.exists():
        print(f"  {synced_path} not found")
        return None

    # Also load original for comparison
    clip_start = entry.get("clip_start", 0)
    clip_end = entry.get("clip_end", entry["duration"])

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_clip = tmp.name
    extract_clip(VIDEO_DIR / entry["file"], clip_start, clip_end, tmp_clip)

    print(f"  Reading frames...")
    orig_frames, fps = read_all_frames(tmp_clip)
    sync_frames, _ = read_all_frames(synced_path)
    n_frames = min(len(orig_frames), len(sync_frames))
    orig_frames = orig_frames[:n_frames]
    sync_frames = sync_frames[:n_frames]
    Path(tmp_clip).unlink(missing_ok=True)

    print(f"  {n_frames} frames at {fps:.1f} fps ({n_frames/fps:.1f}s)")

    # Get mouth mask from first frame with a face
    print(f"  Detecting mouth region...")
    landmarker = create_face_landmarker()
    mouth_mask = None
    for frame in sync_frames[:30]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = get_mouth_mask(rgb, landmarker)
        if mask is not None:
            mouth_mask = mask
            break
    landmarker.close()

    if mouth_mask is None:
        print(f"  No face detected!")
        return None

    mouth_pixels = np.where(mouth_mask)
    n_pixels = len(mouth_pixels[0])
    print(f"  Mouth region: {n_pixels} pixels")

    # Extract brightness time series for each mouth pixel
    # For synced video
    print(f"  Extracting pixel time series (synced)...")
    sync_gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sync_frames])
    sync_series = sync_gray[:, mouth_pixels[0], mouth_pixels[1]]  # (T, N_pixels)

    print(f"  Extracting pixel time series (original)...")
    orig_gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in orig_frames])
    orig_series = orig_gray[:, mouth_pixels[0], mouth_pixels[1]]  # (T, N_pixels)

    # Compute FFT per pixel
    print(f"  Computing FFT ({n_pixels} pixels x {n_frames} frames)...")
    freqs = np.fft.rfftfreq(n_frames, d=1.0 / fps)

    # Frequency bands
    natural_band = (freqs >= 0.5) & (freqs <= 3.0)   # natural lip movement
    wave_band = (freqs >= 5.0) & (freqs <= 15.0)      # artifact oscillation
    high_band = (freqs >= 10.0)                         # very high freq noise

    # Analyze synced video
    sync_fft = np.abs(np.fft.rfft(sync_series.astype(np.float32), axis=0))
    sync_natural_energy = sync_fft[natural_band, :].mean(axis=0)
    sync_wave_energy = sync_fft[wave_band, :].mean(axis=0)
    sync_high_energy = sync_fft[high_band, :].mean(axis=0)

    # Analyze original video
    orig_fft = np.abs(np.fft.rfft(orig_series.astype(np.float32), axis=0))
    orig_natural_energy = orig_fft[natural_band, :].mean(axis=0)
    orig_wave_energy = orig_fft[wave_band, :].mean(axis=0)
    orig_high_energy = orig_fft[high_band, :].mean(axis=0)

    # Wave ratio: how much wave energy relative to natural energy
    sync_wave_ratio = sync_wave_energy / (sync_natural_energy + 1e-6)
    orig_wave_ratio = orig_wave_energy / (orig_natural_energy + 1e-6)

    # Excess wave energy: synced minus original (positive = artifact)
    excess_wave = sync_wave_energy - orig_wave_energy
    excess_ratio = sync_wave_ratio - orig_wave_ratio

    # Per-pixel wave score
    wave_score = np.clip(excess_ratio, 0, None)

    # Statistics
    results = {
        "name": name,
        "n_frames": n_frames,
        "fps": fps,
        "n_mouth_pixels": n_pixels,
        "sync_avg_wave_ratio": float(sync_wave_ratio.mean()),
        "orig_avg_wave_ratio": float(orig_wave_ratio.mean()),
        "excess_wave_ratio": float(excess_ratio.mean()),
        "pct_pixels_with_waves": float((wave_score > 0.1).sum() / n_pixels * 100),
        "max_wave_score": float(wave_score.max()),
        "mean_wave_score": float(wave_score.mean()),
        "sync_wave_energy_mean": float(sync_wave_energy.mean()),
        "orig_wave_energy_mean": float(orig_wave_energy.mean()),
        "sync_high_energy_mean": float(sync_high_energy.mean()),
        "orig_high_energy_mean": float(orig_high_energy.mean()),
        # Frequency spectrum (averaged across all mouth pixels)
        "spectrum": {
            "freqs": freqs.tolist(),
            "sync_avg_magnitude": sync_fft.mean(axis=1).tolist(),
            "orig_avg_magnitude": orig_fft.mean(axis=1).tolist(),
        },
    }

    # Grade
    excess = results["excess_wave_ratio"]
    pct = results["pct_pixels_with_waves"]
    if excess < 0.05 and pct < 10:
        grade = "A"
    elif excess < 0.1 and pct < 25:
        grade = "B"
    elif excess < 0.2 and pct < 50:
        grade = "C"
    else:
        grade = "D"
    results["grade"] = grade

    print(f"\n  === {name} Wave Analysis ===")
    print(f"  Sync wave ratio:  {results['sync_avg_wave_ratio']:.4f}")
    print(f"  Orig wave ratio:  {results['orig_avg_wave_ratio']:.4f}")
    print(f"  Excess (artifact):{results['excess_wave_ratio']:.4f}")
    print(f"  Pixels with waves:{results['pct_pixels_with_waves']:.1f}%")
    print(f"  Max wave score:   {results['max_wave_score']:.4f}")
    print(f"  Grade: {grade}")

    # Save results
    out_dir = VIDEO_DIR / "wave_analysis"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{name}_waves.json", "w") as f:
        json.dump(results, f, indent=2)

    if visualize:
        # Create wave heatmap
        print(f"\n  Creating wave heatmap...")
        heatmap = np.zeros(mouth_mask.shape, dtype=np.float32)
        heatmap[mouth_pixels[0], mouth_pixels[1]] = wave_score
        heatmap = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Overlay on first synced frame
        overlay = sync_frames[0].copy()
        mask3 = np.stack([mouth_mask] * 3, axis=2).astype(np.float32)
        overlay = (overlay * (1 - mask3 * 0.5) + heatmap_color * mask3 * 0.5).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{name}_wave_heatmap.png"), overlay)
        print(f"  Saved: {name}_wave_heatmap.png")

        # Create spectrum plot as text
        print(f"\n  Frequency Spectrum (avg across mouth pixels):")
        print(f"  {'Freq (Hz)':>10} | {'Sync':>10} | {'Orig':>10} | {'Diff':>10}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for i, freq in enumerate(freqs):
            if freq > 15:
                break
            sm = sync_fft[i].mean()
            om = orig_fft[i].mean()
            bar = "+" * min(int((sm - om) / 2), 20) if sm > om else "-" * min(int((om - sm) / 2), 20)
            print(f"  {freq:10.1f} | {sm:10.1f} | {om:10.1f} | {bar}")

    return results


def main():
    p = argparse.ArgumentParser(description="Detect temporal pixel waves in lip-synced video")
    p.add_argument("--name", type=str, help="Person name")
    p.add_argument("--all", action="store_true", help="Analyze all lip-synced members")
    p.add_argument("--visualize", action="store_true", help="Generate heatmap + spectrum")
    args = p.parse_args()

    if args.all:
        with open(ANALYSIS, encoding="utf-8") as f:
            entries = json.load(f)
        results = []
        for entry in entries:
            if entry.get("skip_lipsync"):
                continue
            print(f"\n{'='*60}")
            print(f"  {entry['name']}")
            print(f"{'='*60}")
            r = analyze_waves(entry["name"], visualize=args.visualize)
            if r:
                results.append(r)

        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Name':<12} {'WaveRatio':>10} {'Excess':>10} {'%Pixels':>10} {'Grade':>6}")
        for r in results:
            print(f"  {r['name']:<12} {r['sync_avg_wave_ratio']:>10.4f} "
                  f"{r['excess_wave_ratio']:>10.4f} {r['pct_pixels_with_waves']:>9.1f}% "
                  f"{r['grade']:>6}")
    elif args.name:
        analyze_waves(args.name, visualize=args.visualize)
    else:
        print("Use --name <Name> or --all")


if __name__ == "__main__":
    main()
