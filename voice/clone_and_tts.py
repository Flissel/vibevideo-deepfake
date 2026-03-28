"""
clone_and_tts.py - Quick Demo: Clone voice + generate TTS in one step.
Uses Chatterbox (local, no API key needed).

Usage:
    python clone_and_tts.py voice_sample.wav "Your text here" output.wav
    python clone_and_tts.py voice_sample.wav  # uses default demo text
"""

import sys
from pathlib import Path

from .chatterbox_engine import generate_speech

DEFAULT_TEXT = "Hey, it's your mum, get the shit done."


def main():
    if len(sys.argv) < 2:
        print("Usage: python clone_and_tts.py <reference_audio> [text] [output_path]")
        print("  No API key needed — runs 100% local via Chatterbox.")
        sys.exit(1)

    ref_audio = Path(sys.argv[1])
    text = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEXT
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("tts") / f"{ref_audio.stem}_tts.wav"

    if not ref_audio.exists():
        print(f"Error: Reference audio not found: {ref_audio}")
        sys.exit(1)

    print(f"Reference: {ref_audio}")
    print(f"Text:      {text}")
    print(f"Output:    {output}")
    print(f"Engine:    Chatterbox Turbo (local, no API key)")
    print()

    result = generate_speech(text, ref_audio, output)
    size_kb = result.stat().st_size // 1024
    print(f"\nDone: {result} ({size_kb} KB)")


if __name__ == "__main__":
    main()
