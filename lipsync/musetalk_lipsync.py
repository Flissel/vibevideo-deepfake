"""
musetalk_lipsync.py - MuseTalk-based lip sync (256x256, much sharper than Wav2Lip)

Runs MuseTalk in its own Python 3.10 venv with separate dependencies.
Called from the main pipeline or standalone.

Usage:
  python musetalk_lipsync.py --video input.mp4 --audio tts.mp3 --output output.mp4
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent
MUSETALK_DIR = VIDEO_DIR / "musetalk"
MUSETALK_VENV_PYTHON = MUSETALK_DIR / "venv" / "Scripts" / "python.exe"
MUSETALK_INFERENCE = MUSETALK_DIR / "scripts" / "inference.py"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_musetalk(video_path: Path, audio_path: Path, output_path: Path,
                 version: str = "v15", bbox_shift: int = 0,
                 use_float16: bool = True, batch_size: int = 8,
                 extra_margin: int = 10):
    """Run MuseTalk inference in its own venv."""
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(exist_ok=True)

    # MuseTalk uses os.system with unquoted paths — spaces in filenames break it
    # Copy to temp file without spaces if needed
    import shutil
    tmp_video = None
    if " " in str(video_path):
        tmp_video = MUSETALK_DIR / "_tmp_input_video.mp4"
        shutil.copy2(str(video_path), str(tmp_video))
        video_path = tmp_video

    # Convert audio to WAV if needed (MuseTalk prefers WAV)
    if audio_path.suffix.lower() != ".wav":
        ffmpeg = get_ffmpeg()
        wav_path = output_path.parent / f"_tmp_musetalk_audio.wav"
        subprocess.run([
            ffmpeg, "-y", "-i", str(audio_path),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(wav_path)
        ], capture_output=True, check=True)
        audio_path = wav_path
    else:
        wav_path = None

    # Create temp inference config
    config_path = MUSETALK_DIR / "_tmp_inference_config.yaml"
    result_name = output_path.name
    config_content = (
        f"task_0:\n"
        f" video_path: \"{str(video_path).replace(chr(92), '/')}\"\n"
        f" audio_path: \"{str(audio_path).replace(chr(92), '/')}\"\n"
        f" bbox_shift: {bbox_shift}\n"
        f" result_name: \"{result_name}\"\n"
    )
    config_path.write_text(config_content)

    # Build command
    ffmpeg_dir = str(Path(get_ffmpeg()).parent)
    cmd = [
        str(MUSETALK_VENV_PYTHON),
        str(MUSETALK_INFERENCE),
        "--inference_config", str(config_path),
        "--result_dir", str(output_path.parent / "_musetalk_results"),
        "--version", version,
        "--batch_size", str(batch_size),
        "--ffmpeg_path", ffmpeg_dir,
        "--unet_config", "./models/musetalk/musetalk.json",
    ]
    if use_float16:
        cmd.append("--use_float16")
    if extra_margin:
        cmd.extend(["--extra_margin", str(extra_margin)])

    print(f"  MuseTalk: {video_path.name} + {audio_path.name}")
    print(f"  Version: {version}, float16: {use_float16}, batch: {batch_size}, extra_margin: {extra_margin}")

    # Run in MuseTalk directory (add ffmpeg to PATH for subprocess/shell calls)
    env = {**os.environ, "PYTHONPATH": str(MUSETALK_DIR), "PYTHONIOENCODING": "utf-8"}
    # MuseTalk internally calls ffmpeg via shell — needs it on PATH
    # Add both the imageio_ffmpeg dir AND the MuseTalk dir (has ffmpeg.exe copy)
    env["PATH"] = str(MUSETALK_DIR) + os.pathsep + ffmpeg_dir + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        cmd,
        cwd=str(MUSETALK_DIR),
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"MuseTalk failed with return code {result.returncode}")

    # Find output file
    results_dir = output_path.parent / "_musetalk_results" / version
    candidates = list(results_dir.glob("*.mp4"))
    if not candidates:
        raise RuntimeError(f"No output video found in {results_dir}")

    # Move to final output path
    musetalk_output = candidates[0]
    import shutil
    shutil.move(str(musetalk_output), str(output_path))
    print(f"  MuseTalk output: {output_path.name}")

    # Cleanup
    config_path.unlink(missing_ok=True)
    if wav_path and wav_path.exists():
        wav_path.unlink()
    if tmp_video and tmp_video.exists():
        tmp_video.unlink()
    results_dir_parent = output_path.parent / "_musetalk_results"
    if results_dir_parent.exists():
        shutil.rmtree(results_dir_parent, ignore_errors=True)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="MuseTalk Lip Sync")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--audio", required=True, help="Audio/TTS path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--version", default="v15", choices=["v1", "v15"])
    parser.add_argument("--bbox-shift", type=int, default=0)
    parser.add_argument("--no-float16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = run_musetalk(
        video_path=Path(args.video),
        audio_path=Path(args.audio),
        output_path=Path(args.output),
        version=args.version,
        bbox_shift=args.bbox_shift,
        use_float16=not args.no_float16,
        batch_size=args.batch_size,
    )

    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"\nFertig: {result.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
