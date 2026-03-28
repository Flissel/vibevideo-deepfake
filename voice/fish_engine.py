"""
fish_engine.py - Lokale TTS Engine via Fish Speech OpenAudio S1-mini
Bessere Prosodie als Chatterbox, 80+ Sprachen (DE Tier 2), RLHF-optimiert.
Kein API-Key noetig, laeuft auf RTX 3060 (12 GB VRAM).

Usage:
    from .fish_engine import generate_speech
    generate_speech("Hallo Welt", "ref_audio.wav", "output.wav")
"""

import os
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_engine = None
_model_id = "fishaudio/openaudio-s1-mini"


def get_engine(device: str = None, precision: str = None):
    """Lazy-load Fish Speech S1-mini engine (LLM + DAC codec)."""
    global _engine
    if _engine is not None:
        return _engine

    if device is None:
        device = os.environ.get("FISH_DEVICE", "cuda")
    if precision is None:
        precision = os.environ.get("FISH_PRECISION", "bfloat16")

    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = precision_map.get(precision, torch.bfloat16)

    logger.info(f"Loading Fish Speech S1-mini on {device} ({precision})...")

    # Download model from HuggingFace
    from huggingface_hub import snapshot_download
    model_dir = Path(snapshot_download(_model_id))
    logger.info(f"Model dir: {model_dir}")

    # 1. Load DAC codec (audio decoder)
    from fish_speech.models.dac.inference import load_model as load_dac
    decoder = load_dac("modded_dac_vq", str(model_dir / "codec.pth"), device=device)

    # 2. Launch LLM worker thread (text → semantic tokens)
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(model_dir / "model.pth"),
        device=device,
        precision=dtype,
        compile=False,
    )

    # 3. Create TTS inference engine
    from fish_speech.inference_engine import TTSInferenceEngine
    _engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder,
        precision=dtype,
        compile=False,
    )

    logger.info("Fish Speech S1-mini loaded.")
    return _engine


def unload():
    """Unload model to free VRAM (for switching between engines)."""
    global _engine
    if _engine is not None:
        _engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Fish Speech model unloaded.")


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
        reference_audio: Path to 10-30s reference WAV
        output_path: Where to save the output WAV
        device: "cuda" or "cpu"

    Returns:
        Path to the generated audio file
    """
    import soundfile as sf
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    reference_audio = Path(reference_audio)
    output_path = Path(output_path)

    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    engine = get_engine(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read reference audio bytes
    ref_bytes = reference_audio.read_bytes()

    # Build request
    request = ServeTTSRequest(
        text=text,
        references=[
            ServeReferenceAudio(audio=ref_bytes, text=""),
        ],
        format="wav",
        normalize=True,
        max_new_tokens=1024,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.1,
    )

    logger.info(f"Generating TTS (Fish S1-mini): {text[:60]}... (ref: {reference_audio.name})")

    # Run inference
    result = None
    for chunk in engine.inference(request):
        if chunk.code == "error":
            raise chunk.error or RuntimeError("Fish Speech inference failed")
        if chunk.code == "final":
            result = chunk

    if result is None or result.audio is None:
        raise RuntimeError("No audio generated")

    sample_rate, audio_array = result.audio
    sf.write(str(output_path), audio_array, sample_rate)

    logger.info(f"Saved: {output_path} ({output_path.stat().st_size // 1024} KB)")
    return output_path
