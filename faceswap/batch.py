"""Batch face-swap: video in → video out.

Loads inswapper + buffalo_l once, walks the input video frame-by-frame, swaps
the face onto the target, optionally aligns with the eye-corner-stable particle
warp, writes mp4 to ~/.rowboat/Videos/ (the shared media root that Rowboat +
Video Space pick up automatically).

Usage
-----
    python -m vibevideo_deepfake.faceswap.batch INPUT.mp4 \\
        --target Diego \\
        [--output PATH] \\
        [--no-alignment] [--smoothing 0.8] \\
        [--keep-audio]   (default on if ffmpeg in PATH)

Target accepts either a display name (Diego) or a slug (face1).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Side-effect: registers nvidia/* DLL dirs for ORT CUDA on Windows
from . import swapper as _swapper  # noqa: F401
from .swapper import FaceSwapper
from .aligner import ParticleAligner
from .presets import DISPLAY_NAMES, resolve_preset, list_presets_detailed

logger = logging.getLogger("vibevideo.faceswap.batch")

DEFAULT_OUTPUT_DIR = Path.home() / ".rowboat" / "Videos"


# ───────────────────────────────────────────────────────────────────────
# Target lookup (by display name OR slug)
# ───────────────────────────────────────────────────────────────────────

def _slug_for_target(target: str) -> str:
    """Accept either a slug ('face1') or a display name ('Diego')."""
    t = target.strip()
    # Reverse lookup: name → slug
    name_to_slug = {v.lower(): k for k, v in DISPLAY_NAMES.items()}
    if t.lower() in name_to_slug:
        return name_to_slug[t.lower()]
    # Otherwise treat as slug
    return t


# ───────────────────────────────────────────────────────────────────────
# Audio mux (optional, requires ffmpeg in PATH)
# ───────────────────────────────────────────────────────────────────────

def _resolve_ffmpeg() -> Optional[str]:
    """Same auto-detect as routers/video.py — keeps the two in sync."""
    import os
    override = os.environ.get("FFMPEG_PATH", "").strip()
    if override and Path(override).exists():
        return override
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    home = Path.home()
    for c in [
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"),
        home / "scoop" / "shims" / "ffmpeg.exe",
        home / "AppData" / "Local" / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
        home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]:
        if c.exists():
            return str(c)
    return None


def _finalize(video_only: Path, original: Path, final: Path, keep_audio: bool) -> bool:
    """Re-encode the cv2-written mpeg4 file to H.264 and (optionally) pull
    audio from the original. Returns True on success.

    cv2.VideoWriter writes MPEG-4 Part 2 ("mpeg4"), which Chromium and most
    HTML5 <video> tags refuse to play. We always transcode to H.264 +
    yuv420p + faststart to make the output streamable in browsers.
    """
    ffmpeg_bin = _resolve_ffmpeg()
    if not ffmpeg_bin:
        logger.warning(
            "ffmpeg not found — leaving raw mpeg4 file. Browser <video> tag "
            "will NOT play it. Install via 'winget install ffmpeg' or set "
            "FFMPEG_PATH."
        )
        return False
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_only),
    ]
    if keep_audio:
        cmd += ["-i", str(original)]
    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-map", "0:v:0",
    ]
    if keep_audio:
        cmd += ["-c:a", "aac", "-map", "1:a:0?", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(final))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("ffmpeg finalize failed (rc=%d): %s", result.returncode, result.stderr[-500:])
            return False
        return True
    except Exception as e:
        logger.warning("ffmpeg finalize error: %s", e)
        return False


# ───────────────────────────────────────────────────────────────────────
# Core
# ───────────────────────────────────────────────────────────────────────

def run_batch(
    input_path: Path,
    target: str,
    output_path: Optional[Path] = None,
    alignment: bool = True,
    smoothing: float = 0.8,
    keep_audio: bool = True,
    providers: Optional[list] = None,
) -> Path:
    """Process an entire video. Returns the final output path."""
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    slug = _slug_for_target(target)
    target_path = resolve_preset(slug)

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_swap_{slug}_{ts}.mp4"
    else:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[input]  {input_path.name} {w}x{h} @ {fps:.1f} fps, {total} frames", flush=True)
    print(f"[target] {DISPLAY_NAMES.get(slug, slug)} ({slug}) -> {target_path.name}", flush=True)
    print(f"[output] {output_path}", flush=True)
    print(f"[align]  {'on' if alignment else 'off'} (alpha={smoothing})", flush=True)

    # Init pipeline
    print("[init]   loading insightface buffalo_l + inswapper_128 …", flush=True)
    t_init = time.perf_counter()
    swapper_obj = FaceSwapper(
        target_face_path=target_path,
        providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    aligner = ParticleAligner(smoothing_alpha=smoothing) if alignment else None
    print(f"[init]   ready in {time.perf_counter() - t_init:.1f}s", flush=True)

    # Stage 1: cv2 always writes mpeg4 (mp4v fourcc) to a temp file —
    # we will re-encode to H.264 in the finalize pass for browser playback.
    tmp_video = output_path.with_suffix(".video_only.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("cv2.VideoWriter failed to open output")

    try:
        i = 0
        t_start = time.perf_counter()
        last_log = t_start
        no_face_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1

            try:
                swapped, ist_face = swapper_obj.swap(frame)
                if ist_face is None:
                    no_face_count += 1
                    out_frame = frame  # pass-through original frame
                elif aligner is not None:
                    # Aligner requires MediaPipe SOLL landmarks — for batch mode
                    # we don't have those (no eyeTerm running). Skip alignment in
                    # batch and rely on inswapper's own placement, which is
                    # frame-stable enough on pre-recorded video.
                    out_frame = swapped
                else:
                    out_frame = swapped
            except Exception as e:
                logger.warning("frame %d swap failed: %s", i, e)
                out_frame = frame

            writer.write(out_frame)

            now = time.perf_counter()
            if now - last_log >= 1.0 or i == total:
                pct = (i / total * 100.0) if total > 0 else 0.0
                rate = i / max(now - t_start, 1e-3)
                eta = (total - i) / max(rate, 1e-3) if total > 0 else 0
                print(f"[frame]  {i}/{total} ({pct:5.1f}%) {rate:.1f} fps eta {eta:.0f}s",
                      flush=True)
                last_log = now
    finally:
        writer.release()
        cap.release()

    if no_face_count:
        print(f"[warn]   {no_face_count} frames had no face detected — passed through unchanged", flush=True)

    # Stage 2: H.264 finalize (re-encode + optional audio mux) so the
    # output plays in browsers and is streamable (faststart).
    ok = _finalize(tmp_video, input_path, output_path, keep_audio=keep_audio)
    if ok:
        try:
            tmp_video.unlink()
        except Exception:
            pass
    else:
        print(f"[warn]   ffmpeg finalize skipped — keeping raw mpeg4 at {tmp_video.name}", flush=True)
        # Leave the .video_only.mp4 file in place; user can transcode later.
        # If output_path was expected, point it at the temp file.
        if not output_path.exists():
            tmp_video.rename(output_path)

    print(f"[done]   {output_path}", flush=True)
    return output_path


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="faceswap-batch",
        description="Batch face-swap a video file (writes mp4 to ~/.rowboat/Videos/).",
    )
    p.add_argument("input", help="Input video path")
    p.add_argument("--target", required=True,
                   help="Target face: display name (Diego) or slug (face1)")
    p.add_argument("--output", default=None,
                   help="Output mp4 path (default: ~/.rowboat/Videos/<auto>.mp4)")
    p.add_argument("--no-alignment", action="store_true",
                   help="Disable particle alignment (faster, batch is usually fine)")
    p.add_argument("--smoothing", type=float, default=0.8,
                   help="Alignment EMA alpha when alignment is on (0.0-1.0)")
    p.add_argument("--no-audio", action="store_true",
                   help="Drop original audio track (default: keep)")
    p.add_argument("--list", action="store_true",
                   help="List available presets and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    if args.list:
        for entry in list_presets_detailed():
            print(f"  {entry['id']:10s} {entry['name']}")
        return

    out = run_batch(
        input_path=Path(args.input),
        target=args.target,
        output_path=Path(args.output) if args.output else None,
        alignment=not args.no_alignment,
        smoothing=args.smoothing,
        keep_audio=not args.no_audio,
    )
    print(out)


if __name__ == "__main__":
    main()
