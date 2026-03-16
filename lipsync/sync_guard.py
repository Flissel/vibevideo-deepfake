"""
sync_guard.py  –  Frame-count-anchored audio-video sync enforcement

Core insight: video frame count is ground truth. Audio is flexible.
Always count actual frames, compute duration as n_frames / fps,
and trim/pad audio to match exactly.

Usage:
  python sync_guard.py --verify video.mp4          # check sync
  python sync_guard.py --enforce video.mp4 audio.wav out.mp4  # enforce sync
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


TARGET_FPS = 30
TARGET_SR = 44100
VIDEO_DIR = Path(__file__).parent.parent


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------------------
# Frame counting
# ---------------------------------------------------------------------------

def count_frames(video_path: str | Path) -> int:
    """Count actual frames by iterating through the video with cv2."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        n += 1
    cap.release()
    return n


def get_video_fps(video_path: str | Path) -> float:
    """Get fps from video metadata via cv2."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else TARGET_FPS


def get_audio_duration(video_path: str | Path) -> float:
    """Get audio stream duration using ffmpeg."""
    ffmpeg = get_ffmpeg()
    # Extract audio duration by decoding to null
    result = subprocess.run(
        [ffmpeg, "-i", str(video_path), "-vn", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60
    )
    # Parse duration from stderr
    import re
    # Look for "Duration: HH:MM:SS.ms" in stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if m:
        h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mn * 60 + s + cs / 100.0
    return 0.0


# ---------------------------------------------------------------------------
# Audio trimming / padding
# ---------------------------------------------------------------------------

def trim_audio_to_duration(audio_array: np.ndarray, current_sr: int,
                           target_duration_sec: float,
                           target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Resample (if needed) and trim or zero-pad audio to exact sample count.
    Returns (trimmed_array, n_samples).
    """
    # Resample if sample rates differ
    if current_sr != target_sr:
        from scipy.signal import resample
        n_target = int(len(audio_array) * target_sr / current_sr)
        if audio_array.ndim == 1:
            audio_array = resample(audio_array, n_target)
        else:
            # Multi-channel: resample each channel
            channels = []
            for ch in range(audio_array.shape[1]):
                channels.append(resample(audio_array[:, ch], n_target))
            audio_array = np.column_stack(channels)

    exact_samples = int(round(target_duration_sec * target_sr))

    if len(audio_array) > exact_samples:
        # Trim
        audio_array = audio_array[:exact_samples]
    elif len(audio_array) < exact_samples:
        # Zero-pad
        pad_len = exact_samples - len(audio_array)
        if audio_array.ndim == 1:
            audio_array = np.concatenate([audio_array, np.zeros(pad_len, dtype=audio_array.dtype)])
        else:
            pad = np.zeros((pad_len, audio_array.shape[1]), dtype=audio_array.dtype)
            audio_array = np.concatenate([audio_array, pad])

    return audio_array, exact_samples


def trim_audio_file(audio_path: str | Path, target_duration_sec: float,
                    output_path: str | Path = None) -> Path:
    """
    Trim or pad an audio file to exact duration using ffmpeg.
    If output_path is None, overwrites in place.
    """
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.parent / f"_tmp_trimmed_{audio_path.name}"

    ffmpeg = get_ffmpeg()
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(audio_path),
        "-t", f"{target_duration_sec:.6f}",
        "-acodec", "pcm_s16le", "-ar", str(TARGET_SR),
        str(output_path)
    ], capture_output=True, timeout=30)

    return Path(output_path)


# ---------------------------------------------------------------------------
# Sync enforcement
# ---------------------------------------------------------------------------

def enforce_sync(video_path: str | Path, audio_path: str | Path,
                 output_path: str | Path, fps: float = TARGET_FPS) -> dict:
    """
    Count video frames → compute authoritative duration → mux with
    explicit -t duration (no -shortest).

    Returns dict with sync details.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    n_frames = count_frames(video_path)
    auth_duration = n_frames / fps

    ffmpeg = get_ffmpeg()

    # Mux with explicit duration — no -shortest
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{auth_duration:.6f}",
        "-map", "0:v:0", "-map", "1:a:0",
        str(output_path)
    ], capture_output=True, timeout=120)

    return {
        "n_frames": n_frames,
        "fps": fps,
        "auth_duration_sec": auth_duration,
        "output": str(output_path),
    }


# ---------------------------------------------------------------------------
# Sync verification
# ---------------------------------------------------------------------------

def verify_sync(video_path: str | Path, tolerance_ms: float = 15.0) -> dict:
    """
    Compare video-frame-derived duration vs audio duration.
    Returns dict with is_synced, delta_ms, details.
    """
    video_path = Path(video_path)

    n_frames = count_frames(video_path)
    fps = get_video_fps(video_path)
    video_duration = n_frames / fps

    audio_duration = get_audio_duration(video_path)

    delta_sec = abs(video_duration - audio_duration)
    delta_ms = delta_sec * 1000

    return {
        "is_synced": delta_ms <= tolerance_ms,
        "delta_ms": round(delta_ms, 2),
        "n_frames": n_frames,
        "fps": round(fps, 2),
        "video_duration_sec": round(video_duration, 4),
        "audio_duration_sec": round(audio_duration, 4),
        "file": video_path.name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audio-Video Sync Guard")
    parser.add_argument("--verify", nargs="+", metavar="VIDEO",
                        help="Verify sync of one or more video files")
    parser.add_argument("--enforce", nargs=3, metavar=("VIDEO", "AUDIO", "OUTPUT"),
                        help="Enforce sync: count frames, trim audio, mux")
    parser.add_argument("--tolerance", type=float, default=15.0,
                        help="Sync tolerance in ms (default: 15)")
    args = parser.parse_args()

    if args.verify:
        print("Sync Verification")
        print("=" * 60)
        all_synced = True
        for vpath in args.verify:
            vpath = Path(vpath)
            if not vpath.exists():
                # Try relative to VIDEO_DIR
                vpath = VIDEO_DIR / vpath
            if not vpath.exists():
                print(f"  {vpath.name}: FILE NOT FOUND")
                continue

            result = verify_sync(vpath, tolerance_ms=args.tolerance)
            status = "SYNC OK" if result["is_synced"] else "OUT OF SYNC"
            print(f"  {result['file']}: {status}  "
                  f"(delta={result['delta_ms']:.1f}ms, "
                  f"frames={result['n_frames']}, "
                  f"fps={result['fps']}, "
                  f"video={result['video_duration_sec']:.3f}s, "
                  f"audio={result['audio_duration_sec']:.3f}s)")
            if not result["is_synced"]:
                all_synced = False

        print("=" * 60)
        print("Result:", "ALL SYNCED" if all_synced else "SYNC ISSUES DETECTED")
        sys.exit(0 if all_synced else 1)

    elif args.enforce:
        video, audio, output = args.enforce
        print(f"Enforcing sync: {video} + {audio} → {output}")
        result = enforce_sync(video, audio, output)
        print(f"  Frames: {result['n_frames']}")
        print(f"  Duration: {result['auth_duration_sec']:.4f}s")
        print(f"  Output: {result['output']}")

        # Verify result
        vr = verify_sync(output)
        status = "SYNC OK" if vr["is_synced"] else "OUT OF SYNC"
        print(f"  Verification: {status} (delta={vr['delta_ms']:.1f}ms)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
