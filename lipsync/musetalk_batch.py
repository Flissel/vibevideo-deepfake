"""
musetalk_batch.py - Batch pipeline: Align Audio -> MuseTalk -> LS Blend.

Processes all lip-sync members (skip_lipsync=false) through the full pipeline:
  1. Extract clip segment from original video
  2. Align TTS audio to match original speech rhythm (word-level DTW)
  3. Run MuseTalk (256x256) with aligned audio
  4. Apply LS boundary matching to eliminate seams
  5. Save final to lip_sync/<Name>.mp4

Usage:
  python musetalk_batch.py                    # all members
  python musetalk_batch.py --only Lisa        # single person
  python musetalk_batch.py --skip-align       # skip audio alignment
  python musetalk_batch.py --skip-ls          # skip LS blending
  python musetalk_batch.py --debug            # verbose output
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent
TTS_DIR = VIDEO_DIR / "tts"
OUT_DIR = VIDEO_DIR / "lip_sync"
ANALYSIS = VIDEO_DIR / "analysis.json"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_analysis():
    with open(ANALYSIS, encoding="utf-8") as f:
        return json.load(f)


def extract_clip(entry: dict, out_path: Path):
    """Extract the clip segment from the original video."""
    ffmpeg = get_ffmpeg()
    video = VIDEO_DIR / entry["file"]
    if not video.exists():
        video = VIDEO_DIR / "data" / entry["file"]
    start = entry.get("clip_start", 0)
    end = entry.get("clip_end", entry["duration"])

    cmd = [
        ffmpeg, "-y",
        "-i", str(video),
        "-ss", str(start),
        "-t", str(end - start),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def process_person(entry: dict, skip_align: bool = False,
                   skip_ls: bool = False, debug: bool = False):
    """Full pipeline for one person."""
    name = entry["name"]
    tts_path = TTS_DIR / f"{name}.mp3"

    if not tts_path.exists():
        print(f"  WARNING: TTS file not found: {tts_path.name}, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    OUT_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Extract clip
        print(f"\n  [1/4] Extracting clip...")
        clip_path = tmpdir / f"{name}_clip.mp4"
        extract_clip(entry, clip_path)
        clip_size = clip_path.stat().st_size / (1024 * 1024)
        print(f"    Clip: {clip_size:.1f} MB")

        # Step 2: Align TTS audio
        if skip_align:
            print(f"\n  [2/4] Audio alignment SKIPPED")
            audio_path = tts_path
        else:
            print(f"\n  [2/4] Aligning TTS audio to original rhythm...")
            from align_audio import align_tts_to_original
            aligned_path = tmpdir / f"{name}_aligned.wav"
            align_tts_to_original(
                original_path=(VIDEO_DIR / entry["file"] if (VIDEO_DIR / entry["file"]).exists()
                              else VIDEO_DIR / "data" / entry["file"]),
                tts_path=tts_path,
                output_path=aligned_path,
                clip_start=entry.get("clip_start", 0),
                clip_end=entry.get("clip_end", None),
                debug=debug,
            )
            audio_path = aligned_path

        # Step 3: MuseTalk
        print(f"\n  [3/4] Running MuseTalk (256x256)...")
        from musetalk_lipsync import run_musetalk
        musetalk_output = tmpdir / f"{name}_musetalk.mp4"
        run_musetalk(
            video_path=clip_path,
            audio_path=audio_path,
            output_path=musetalk_output,
            version="v15",
            use_float16=True,
            batch_size=8,
        )

        # Step 4: LS boundary matching
        if skip_ls:
            print(f"\n  [4/4] LS blending SKIPPED")
            final_output = musetalk_output
        else:
            print(f"\n  [4/4] Applying LS boundary matching...")
            from ls_blend import process_video as ls_process
            ls_output = tmpdir / f"{name}_ls.mp4"
            ls_process(
                original_path=str(clip_path),
                generated_path=str(musetalk_output),
                output_path=str(ls_output),
            )
            final_output = ls_output

        # Copy to final output
        dest = OUT_DIR / f"{name}.mp4"
        shutil.copy2(str(final_output), str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"\n  DONE: {dest.name} ({size_mb:.1f} MB)")

    return dest


def main():
    p = argparse.ArgumentParser(
        description="Batch MuseTalk pipeline with audio alignment + LS blending"
    )
    p.add_argument("--only", type=str, help="Process only this person")
    p.add_argument("--skip-align", action="store_true",
                   help="Skip audio alignment step")
    p.add_argument("--skip-ls", action="store_true",
                   help="Skip LS boundary matching step")
    p.add_argument("--debug", action="store_true",
                   help="Verbose debug output")
    args = p.parse_args()

    entries = load_analysis()

    # Filter to lip-sync members only
    to_process = []
    for entry in entries:
        if entry.get("skip_lipsync"):
            print(f"  Skip: {entry['name']} (skip_lipsync=true)")
            continue
        if args.only and entry["name"].lower() != args.only.lower():
            continue
        to_process.append(entry)

    if not to_process:
        print("No members to process.")
        return

    print(f"\nPipeline: {'Align -> ' if not args.skip_align else ''}"
          f"MuseTalk -> "
          f"{'LS Blend' if not args.skip_ls else 'Done'}")
    print(f"Members: {', '.join(e['name'] for e in to_process)}")

    results = []
    for entry in to_process:
        result = process_person(
            entry,
            skip_align=args.skip_align,
            skip_ls=args.skip_ls,
            debug=args.debug,
        )
        if result:
            results.append(result)

    print(f"\n{'='*60}")
    print(f"  All done! {len(results)}/{len(to_process)} processed")
    print(f"{'='*60}")
    for r in results:
        size_mb = r.stat().st_size / (1024 * 1024)
        print(f"  {r.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
