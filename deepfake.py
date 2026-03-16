#!/usr/bin/env python3
"""
deepfake - CLI for VibeMind Lip Sync & Voice Cloning Tools

Usage:
  python deepfake.py lipsync <command>   Lip sync tools
  python deepfake.py voice <command>     Voice cloning & TTS

Examples:
  python deepfake.py lipsync run --only Surya
  python deepfake.py lipsync sweep --person Surya
  python deepfake.py lipsync analyze
  python deepfake.py voice clone
  python deepfake.py voice tts
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def run_script(script_path: str, args: list[str] = None):
    """Run a Python script with optional arguments."""
    full_path = BASE_DIR / script_path
    if not full_path.exists():
        print(f"ERROR: Script not found: {full_path}")
        sys.exit(1)
    cmd = [sys.executable, str(full_path)] + (args or [])
    return subprocess.run(cmd, cwd=str(BASE_DIR))


# ============================================================
# Lip Sync Tools
# ============================================================

LIPSYNC_COMMANDS = {
    "run":      ("lipsync/musetalk_batch.py",          "Run MuseTalk lip sync"),
    "wav2lip":  ("lipsync/lip_sync.py",                "Run Wav2Lip lip sync"),
    "blend":    ("lipsync/adaptive_blend.py",           "Adaptive diff-driven blending"),
    "sweep":    ("lipsync/quality/auto_sweep.py",      "Parameter sweep (find best settings)"),
    "analyze":  ("lipsync/quality/deep_analysis.py",   "Deep quality analysis"),
    "mouth":    ("lipsync/quality/mouth_analyze.py",   "Mouth region diff analysis"),
    "waves":    ("lipsync/quality/wave_detector.py",   "Detect wave artifacts"),
    "test":     ("lipsync/quality/test_enhancements.py", "Test blend enhancements"),
    "align":    ("lipsync/align_audio.py",             "DTW audio alignment"),
    "sync":     ("lipsync/sync_guard.py",              "A/V sync enforcement"),
}


def cmd_lipsync(args):
    sub = args.sub
    extra = args.extra

    if sub == "help" or sub not in LIPSYNC_COMMANDS:
        print("\nLip Sync Tools:\n")
        for name, (_, desc) in LIPSYNC_COMMANDS.items():
            print(f"  {name:10s} {desc}")
        print(f"\nUsage: python deepfake.py lipsync <command> [args...]")
        return

    script, desc = LIPSYNC_COMMANDS[sub]
    print(f"\n--- {sub}: {desc} ---\n")
    result = run_script(script, extra)
    sys.exit(result.returncode)


# ============================================================
# Voice Tools
# ============================================================

VOICE_COMMANDS = {
    "clone":       ("voice/clone_voices.py",       "Clone voices via ElevenLabs"),
    "tts":         ("voice/generate_tts.py",       "Generate TTS voiceover per person"),
    "transcripts": ("voice/export_transcripts.py", "Export editable transcripts"),
    "quick":       ("voice/clone_and_tts.py",      "Quick clone + TTS from audio"),
}


def cmd_voice(args):
    sub = args.sub
    extra = args.extra

    if sub == "help" or sub not in VOICE_COMMANDS:
        print("\nVoice Tools:\n")
        for name, (_, desc) in VOICE_COMMANDS.items():
            print(f"  {name:14s} {desc}")
        print(f"\nUsage: python deepfake.py voice <command> [args...]")
        return

    script, desc = VOICE_COMMANDS[sub]
    print(f"\n--- {sub}: {desc} ---\n")
    result = run_script(script, extra)
    sys.exit(result.returncode)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="deepfake",
        description="VibeMind Deepfake Tools - Lip Sync & Voice Cloning (PRIVATE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Workflows:
  lipsync    Lip sync tools and quality analysis
  voice      Voice cloning and TTS generation

Examples:
  python deepfake.py lipsync run --only Surya
  python deepfake.py lipsync sweep --person Surya
  python deepfake.py lipsync analyze
  python deepfake.py voice clone
  python deepfake.py voice tts""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # lipsync
    p_lipsync = subparsers.add_parser("lipsync", help="Lip sync tools")
    p_lipsync.add_argument("sub", nargs="?", default="help",
                           help="Command (run, wav2lip, blend, sweep, analyze, mouth, waves, test, align, sync)")
    p_lipsync.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args")

    # voice
    p_voice = subparsers.add_parser("voice", help="Voice cloning & TTS")
    p_voice.add_argument("sub", nargs="?", default="help",
                         help="Command (clone, tts, transcripts, quick)")
    p_voice.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args")

    args = parser.parse_args()

    if args.command == "lipsync":
        cmd_lipsync(args)
    elif args.command == "voice":
        cmd_voice(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
