"""
tts_engine.py - Unified TTS Dispatcher
Waehlt automatisch die beste verfuegbare Engine:
  - "fish":       Fish Speech S1-mini (500M, RLHF, 80+ Sprachen, beste Prosodie)
  - "chatterbox": Chatterbox Turbo (350M, schnell, expressive Tags)
  - "auto":       Fish falls verfuegbar, sonst Chatterbox

Env: TTS_ENGINE=auto|fish|chatterbox (default: auto)

Usage:
    from .tts_engine import generate_speech
    generate_speech("Hallo Welt", "ref.wav", "output.wav")
    generate_speech("Hallo Welt", "ref.wav", "output.wav", engine="fish")
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _fish_available() -> bool:
    """Check if Fish Speech is installed and importable."""
    try:
        from fish_speech.inference_engine import TTSInferenceEngine  # noqa: F401
        return True
    except ImportError:
        return False


def generate_speech(
    text: str,
    reference_audio: str | Path,
    output_path: str | Path,
    engine: str = None,
    device: str = None,
) -> Path:
    """
    Generate speech with voice cloning.

    Args:
        text: Text to synthesize
        reference_audio: Path to reference WAV (10-30s)
        output_path: Where to save the output WAV
        engine: "fish", "chatterbox", or "auto" (default from TTS_ENGINE env)
        device: "cuda" or "cpu"

    Returns:
        Path to the generated audio file
    """
    if engine is None:
        engine = os.environ.get("TTS_ENGINE", "auto")

    if engine == "auto":
        engine = "fish" if _fish_available() else "chatterbox"
        logger.info(f"TTS engine auto-selected: {engine}")

    if engine == "fish":
        from .fish_engine import generate_speech as fish_generate
        return fish_generate(text, reference_audio, output_path, device)
    elif engine == "chatterbox":
        from .chatterbox_engine import generate_speech as cb_generate
        return cb_generate(text, reference_audio, output_path, device)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}. Use 'fish', 'chatterbox', or 'auto'.")
