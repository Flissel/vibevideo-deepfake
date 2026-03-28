"""
chatterbox_engine.py - Lokale Voice-Clone + TTS Engine via Chatterbox
Ersetzt ElevenLabs komplett. Kein API-Key noetig.

Chatterbox braucht kein separates Cloning — das Reference-Audio wird direkt
beim Generate mitgegeben. Das vereinfacht die Pipeline erheblich.

Usage:
    from .chatterbox_engine import generate_speech
    generate_speech("Hallo Welt", "ref_audio.wav", "output.wav")
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None


def get_model(device: str = None):
    """Lazy-load Chatterbox Turbo model (350M params)."""
    global _model
    if _model is None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        if device is None:
            device = os.environ.get("CHATTERBOX_DEVICE", "cuda")
        logger.info(f"Loading Chatterbox Turbo on {device}...")
        _model = ChatterboxTurboTTS.from_pretrained(device=device)
        logger.info("Chatterbox Turbo loaded.")
    return _model


def generate_speech(
    text: str,
    reference_audio: str | Path,
    output_path: str | Path,
    device: str = None,
) -> Path:
    """
    Generate speech cloning the voice from reference_audio.

    Args:
        text: Text to synthesize
        reference_audio: Path to 10-15s reference WAV/MP3
        output_path: Where to save the output WAV
        device: "cuda" or "cpu" (default from env CHATTERBOX_DEVICE)

    Returns:
        Path to the generated audio file
    """
    import torchaudio as ta

    reference_audio = Path(reference_audio)
    output_path = Path(output_path)

    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    model = get_model(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating TTS: {text[:60]}... (ref: {reference_audio.name})")
    wav = model.generate(str(text), audio_prompt_path=str(reference_audio))
    ta.save(str(output_path), wav, model.sr)

    logger.info(f"Saved: {output_path} ({output_path.stat().st_size // 1024} KB)")
    return output_path
