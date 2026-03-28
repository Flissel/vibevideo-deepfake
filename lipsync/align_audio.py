"""
align_audio.py - Word-level TTS-to-original audio alignment.

Uses faster-whisper to get word timestamps from both original video audio
and TTS audio, matches words via DTW, then adjusts ONLY the silence gaps
between words (never stretches speech) so timing matches the original.

This preserves 100% of the TTS voice quality — only pauses change.

Usage:
  python align_audio.py --original original.mp4 --tts tts.mp3 --output aligned.wav
  python align_audio.py --original original.mp4 --tts tts.mp3 --output aligned.wav --debug
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


VIDEO_DIR = Path(__file__).parent.parent


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio_to_wav(input_path: Path, output_wav: Path, sr: int = 16000):
    """Extract audio from video/audio file to 16kHz mono WAV."""
    ffmpeg = get_ffmpeg()
    subprocess.run([
        ffmpeg, "-y", "-i", str(input_path),
        "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        str(output_wav),
    ], capture_output=True, check=True)
    return output_wav


def whisper_timestamps(audio_path: Path, language: str = "en"):
    """Get word-level timestamps using faster-whisper.

    Returns list of (word, start_sec, end_sec).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append((w.word.strip(), w.start, w.end))
    return words


def dtw_match_words(orig_words, tts_words):
    """Match TTS words to original words using simple DTW on word text similarity.

    Returns list of (orig_idx, tts_idx) pairs.
    """
    n = len(orig_words)
    m = len(tts_words)
    if n == 0 or m == 0:
        return []

    # Cost matrix: 0 if words match (case-insensitive), 1 otherwise
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ow = orig_words[i - 1][0].lower().strip(".,!?;:'\"")
            tw = tts_words[j - 1][0].lower().strip(".,!?;:'\"")
            # Levenshtein-like: 0 if exact match, partial penalty otherwise
            if ow == tw:
                c = 0.0
            elif ow.startswith(tw) or tw.startswith(ow):
                c = 0.3
            else:
                c = 1.0
            cost[i, j] = c + min(cost[i - 1, j - 1],  # match
                                  cost[i - 1, j] + 0.5,  # skip orig word
                                  cost[i, j - 1] + 0.5)  # skip tts word

    # Backtrace
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        candidates = [
            (cost[i - 1, j - 1], "match"),
            (cost[i - 1, j], "skip_orig"),
            (cost[i, j - 1], "skip_tts"),
        ]
        best = min(candidates, key=lambda x: x[0])
        if best[1] == "match":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif best[1] == "skip_orig":
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


def silence_align_segments(tts_audio: np.ndarray, sr: int,
                           tts_words: list, orig_words: list,
                           pairs: list,
                           target_duration: float = None) -> np.ndarray:
    """Align TTS to original by adjusting ONLY silence gaps between words.

    Speech segments are copied bit-perfect (no stretching).
    Only the gaps (silence) between words are expanded or shrunk to match
    the original speaker's timing/rhythm.

    Args:
        target_duration: if set, trim/pad output to match this duration exactly.

    This preserves 100% of the TTS voice quality.
    """
    if not pairs:
        return tts_audio

    # Build list of TTS speech chunks with their target start times
    # Each entry: (tts_audio_start, tts_audio_end, target_start_time)
    chunks = []

    for idx, (oi, ti) in enumerate(pairs):
        tts_word_start = tts_words[ti][1]
        tts_word_end = tts_words[ti][2]
        orig_word_start = orig_words[oi][1]

        # For first word, include any audio before it (leading silence/breath)
        if idx == 0:
            audio_start = 0.0
            # Offset: how much leading audio before the first word
            leading = tts_word_start
            target_start = max(0.0, orig_word_start - leading)
        else:
            # Start from end of previous TTS word (include the speech only)
            prev_ti = pairs[idx - 1][1]
            prev_tts_end = tts_words[prev_ti][2]
            audio_start = prev_tts_end
            target_start = orig_word_start - (tts_word_start - audio_start)

        # End at this word's end
        audio_end = tts_word_end

        # For last word, include trailing audio
        if idx == len(pairs) - 1:
            audio_end = len(tts_audio) / sr

        chunks.append((audio_start, audio_end, target_start))

    # Now build output: place each chunk at its target time with silence gaps
    # Determine total output duration
    last_chunk = chunks[-1]
    last_chunk_dur = last_chunk[1] - last_chunk[0]
    total_dur = last_chunk[2] + last_chunk_dur + 0.05  # small tail
    total_samples = int(total_dur * sr)
    output = np.zeros(total_samples, dtype=np.float32)

    n_adjusted = 0
    for audio_start, audio_end, target_start in chunks:
        s_sample = int(audio_start * sr)
        e_sample = int(audio_end * sr)
        e_sample = min(e_sample, len(tts_audio))
        if e_sample <= s_sample:
            continue

        chunk = tts_audio[s_sample:e_sample].astype(np.float32)
        out_start = int(max(0, target_start) * sr)
        out_end = out_start + len(chunk)

        # Extend output if needed
        if out_end > len(output):
            output = np.pad(output, (0, out_end - len(output)))

        # Crossfade to avoid clicks at boundaries (5ms fade)
        fade_samples = min(int(0.005 * sr), len(chunk) // 4, 80)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
            chunk[:fade_samples] *= fade_in
            chunk[-fade_samples:] *= fade_out
            # Also fade existing audio at the insertion point
            if out_start > 0:
                existing_end = min(out_start + fade_samples, len(output))
                fade_len = existing_end - out_start
                output[out_start:existing_end] *= np.linspace(
                    1, 0, fade_len, dtype=np.float32)

        # Mix in (additive for crossfade region, overwrite for rest)
        out_end = min(out_end, len(output))
        output[out_start:out_end] += chunk[:out_end - out_start]
        n_adjusted += 1

    # Trim to target duration if specified
    if target_duration is not None:
        target_samples = int(target_duration * sr)
        if len(output) > target_samples:
            output = output[:target_samples]
        elif len(output) < target_samples:
            output = np.pad(output, (0, target_samples - len(output)))
    else:
        # Trim trailing silence
        nonzero = np.nonzero(np.abs(output) > 1e-6)[0]
        if len(nonzero) > 0:
            last_nonzero = np.max(nonzero)
            output = output[:last_nonzero + int(0.05 * sr)]

    print(f"    Placed {n_adjusted} speech chunks, silence-only alignment")
    print(f"    Output duration: {len(output)/sr:.2f}s"
          + (f" (trimmed to {target_duration:.2f}s)" if target_duration else ""))

    return output


def align_tts_to_original(original_path: Path, tts_path: Path,
                          output_path: Path, clip_start: float = 0,
                          clip_end: float = None, debug: bool = False):
    """Full alignment pipeline: extract audio, get timestamps, match, stretch."""
    sr = 16000

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract original audio (clip segment only)
        orig_wav = tmpdir / "original.wav"
        ffmpeg = get_ffmpeg()
        cmd = [ffmpeg, "-y", "-i", str(original_path)]
        if clip_start > 0:
            cmd.extend(["-ss", str(clip_start)])
        if clip_end is not None:
            cmd.extend(["-t", str(clip_end - clip_start)])
        cmd.extend(["-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
                     str(orig_wav)])
        subprocess.run(cmd, capture_output=True, check=True)

        # Extract TTS audio
        tts_wav = tmpdir / "tts.wav"
        extract_audio_to_wav(tts_path, tts_wav, sr)

        # Get word timestamps
        print("  Getting original word timestamps...")
        orig_words = whisper_timestamps(orig_wav)
        print(f"    {len(orig_words)} words found")

        print("  Getting TTS word timestamps...")
        tts_words = whisper_timestamps(tts_wav)
        print(f"    {len(tts_words)} words found")

        if debug:
            print("\n  Original words:")
            for w, s, e in orig_words[:10]:
                print(f"    [{s:.2f}-{e:.2f}] {w}")
            print("\n  TTS words:")
            for w, s, e in tts_words[:10]:
                print(f"    [{s:.2f}-{e:.2f}] {w}")

        # Match words
        print("  Matching words (DTW)...")
        pairs = dtw_match_words(orig_words, tts_words)
        print(f"    {len(pairs)} word pairs matched")

        if debug and pairs:
            print("\n  Matched pairs (first 10):")
            for oi, ti in pairs[:10]:
                ow = orig_words[oi]
                tw = tts_words[ti]
                print(f"    '{ow[0]}' [{ow[1]:.2f}s] <-> '{tw[0]}' [{tw[1]:.2f}s]")

        # Load TTS audio for stretching
        tts_audio, _ = librosa.load(str(tts_wav), sr=sr, mono=True)

        # Get original clip duration for trimming
        orig_audio, _ = librosa.load(str(orig_wav), sr=sr, mono=True)
        orig_duration = len(orig_audio) / sr

        # Silence-only alignment (no stretching — preserves voice quality)
        print("  Aligning by adjusting silences only (no voice stretching)...")
        aligned = silence_align_segments(tts_audio, sr, tts_words, orig_words,
                                         pairs, target_duration=orig_duration)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), aligned, sr)

        dur_orig = len(librosa.load(str(orig_wav), sr=sr, mono=True)[0]) / sr
        dur_aligned = len(aligned) / sr
        print(f"  Original duration: {dur_orig:.2f}s")
        print(f"  Aligned TTS duration: {dur_aligned:.2f}s")
        print(f"  Saved: {output_path}")

    return output_path


def main():
    p = argparse.ArgumentParser(description="Word-level TTS-to-original alignment")
    p.add_argument("--original", required=True, help="Original video/audio")
    p.add_argument("--tts", required=True, help="TTS audio file")
    p.add_argument("--output", required=True, help="Output aligned WAV")
    p.add_argument("--clip-start", type=float, default=0)
    p.add_argument("--clip-end", type=float, default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    align_tts_to_original(
        Path(args.original), Path(args.tts), Path(args.output),
        clip_start=args.clip_start, clip_end=args.clip_end,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
