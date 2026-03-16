"""
generate_tts.py - TTS Audio mit geklonten ElevenLabs Stimmen generieren
Liest transcripts/<Name>.txt und generiert tts/<Name>.mp3

Voraussetzung:
  - python export_transcripts.py    (Transkripte exportiert)
  - transcripts/*.txt bearbeitet    (Text angepasst)
  - python clone_voices.py          (Voice-IDs in analysis.json)

Verwendung:
  python generate_tts.py                     # alle Personen
  python generate_tts.py --only Moritz       # nur eine Person
  python generate_tts.py --overwrite         # bestehende TTS ueberschreiben
"""

import argparse
import json
import os
import sys
from pathlib import Path

VIDEO_DIR  = Path(__file__).parent.parent
TRANS_DIR  = VIDEO_DIR / "transcripts"
TTS_DIR    = VIDEO_DIR / "tts"

# ElevenLabs TTS Einstellungen
VOICE_MODEL    = "eleven_multilingual_v2"   # beste Qualitaet, unterstuetzt DE+EN
VOICE_STABILITY      = 0.5    # 0.0 = expressiver, 1.0 = stabiler
VOICE_SIMILARITY     = 0.85   # wie nah am Original-Klon
VOICE_STYLE          = 0.2    # Stil-Exaggeration (0 = neutral)
VOICE_SPEAKER_BOOST  = True   # Aehnlichkeit zum Klon verstaerken


def load_analysis() -> list:
    with open(VIDEO_DIR / "analysis.json", encoding="utf-8") as f:
        return json.load(f)


def get_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        env_file = VIDEO_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ELEVENLABS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
                    break
    if not key:
        print("Fehler: ELEVENLABS_API_KEY nicht gefunden.")
        print("  Erstelle .env Datei mit: ELEVENLABS_API_KEY=sk_...")
        sys.exit(1)
    return key


def generate_tts(client, voice_id: str, text: str, out_path: Path):
    """Generiert TTS-Audio via Stream und speichert als MP3."""
    from elevenlabs import VoiceSettings

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in client.text_to_speech.stream(
            voice_id=voice_id,
            text=text,
            model_id=VOICE_MODEL,
            voice_settings=VoiceSettings(
                stability=VOICE_STABILITY,
                similarity_boost=VOICE_SIMILARITY,
                style=VOICE_STYLE,
                use_speaker_boost=VOICE_SPEAKER_BOOST,
            ),
            output_format="mp3_44100_128",
        ):
            if chunk:
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser(description="TTS Audio generieren")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Nur bestimmte Personen verarbeiten")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bestehende TTS-Dateien ueberschreiben")
    args = parser.parse_args()

    api_key  = get_api_key()
    analysis = load_analysis()

    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)

    TTS_DIR.mkdir(exist_ok=True)

    print("\nGeneriere TTS Audio...\n")

    for entry in analysis:
        name     = entry["name"]
        voice_id = entry.get("voice_id")

        if args.only and name not in args.only:
            continue

        print(f"[{name}]")

        if not voice_id:
            print(f"  Kein voice_id in analysis.json. Zuerst python clone_voices.py ausfuehren.\n")
            continue

        # Transkript-Datei lesen
        txt_file = TRANS_DIR / f"{name}.txt"
        if not txt_file.exists():
            print(f"  Keine transcripts/{name}.txt gefunden.")
            print(f"  Fuehre zuerst python export_transcripts.py aus.\n")
            continue

        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  transcripts/{name}.txt ist leer, wird uebersprungen.\n")
            continue

        out_path = TTS_DIR / f"{name}.mp3"

        if out_path.exists() and not args.overwrite:
            print(f"  tts/{name}.mp3 existiert bereits (--overwrite zum Ueberschreiben)\n")
            continue

        print(f"  Voice-ID:  {voice_id}")
        print(f"  Text:      {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Generiere tts/{name}.mp3...")

        try:
            generate_tts(client, voice_id, text, out_path)
            size_kb = out_path.stat().st_size // 1024
            print(f"  Gespeichert: tts/{name}.mp3 ({size_kb} KB)\n")
        except Exception as e:
            print(f"  Fehler: {e}\n")

    # Zusammenfassung
    tts_files = list(TTS_DIR.glob("*.mp3"))
    print(f"TTS fertig: {len(tts_files)} Datei(en) in tts/")
    for f in sorted(tts_files):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    print("\nNaechster Schritt:")
    print("  python build_video.py  ->  Video mit TTS-Stimmen")


if __name__ == "__main__":
    main()
