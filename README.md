# vibevideo-deepfake

Private lip sync and voice cloning tools for VibeMind team video production.

> **PRIVATE REPOSITORY** — Contains deepfake tools. Do not share or make public.
> All persons depicted have given explicit consent.
> See [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md).

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add ELEVENLABS_API_KEY

# Lip sync
python deepfake.py lipsync run --only Surya

# Voice cloning
python deepfake.py voice clone
python deepfake.py voice tts
```

## Commands

### Lip Sync

| Command | Description |
| ------- | ----------- |
| `lipsync run` | Run MuseTalk lip sync (batch) |
| `lipsync wav2lip` | Run Wav2Lip lip sync |
| `lipsync blend` | Adaptive diff-driven blending |
| `lipsync sweep` | Parameter sweep (find best settings) |
| `lipsync analyze` | Deep quality analysis |
| `lipsync mouth` | Mouth region diff analysis |
| `lipsync waves` | Detect wave artifacts |
| `lipsync align` | DTW audio alignment |
| `lipsync sync` | A/V sync enforcement |

### Voice

| Command | Description |
| ------- | ----------- |
| `voice clone` | Clone voices via ElevenLabs |
| `voice tts` | Generate TTS voiceover per person |
| `voice transcripts` | Export editable transcripts |
| `voice quick` | Quick clone + TTS from audio |

## Project Structure

```
deepfake.py           # CLI entry point
lipsync/              # Lip sync engines + blending
  quality/            # Analysis & parameter tuning
voice/                # Voice cloning + TTS
analysis.json         # Team metadata (names, roles, voice IDs)
```

## Tech Stack

- **Lip Sync**: MuseTalk (256x256), Wav2Lip
- **Voice**: ElevenLabs (Cloning, TTS, Scribe STT)
- **Blending**: Poisson, Least-Squares, Adaptive Diff-Driven
- **Audio**: DTW alignment, A/V sync enforcement
- **Quality**: Pixel/temporal/frequency analysis, wave artifact detection

## API Keys Required

| Key | Service |
| --- | ------- |
| `ELEVENLABS_API_KEY` | ElevenLabs (voice cloning, TTS) |

## Responsible AI

See [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md) for our AI ethics policy.
All team members have provided explicit consent for voice cloning and lip sync.

## License

Proprietary — see [LICENSE](LICENSE)
