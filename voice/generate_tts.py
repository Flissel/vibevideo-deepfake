"""
generate_tts.py - TTS Audio mit Chatterbox Voice Cloning generieren
Liest transcripts/<Name>.txt und generiert tts/<Name>.wav

Voraussetzung:
  - python export_transcripts.py    (Transkripte exportiert)
  - transcripts/*.txt bearbeitet    (Text angepasst)
  - python clone_voices.py          (Reference-Audio in voice_references/)

Verwendung:
  python generate_tts.py                     # alle Personen
  python generate_tts.py --only Moritz       # nur eine Person
  python generate_tts.py --overwrite         # bestehende TTS ueberschreiben
"""

import argparse
import json
from pathlib import Path

from .chatterbox_engine import generate_speech

VIDEO_DIR  = Path(__file__).parent.parent
ROWBOAT_VIDEOS = Path.home() / ".rowboat" / "Videos"
TRANS_DIR  = VIDEO_DIR / "transcripts"
TTS_DIR    = ROWBOAT_VIDEOS / "tts"
REF_DIR    = ROWBOAT_VIDEOS / "voice_references"


def load_analysis() -> list:
    with open(VIDEO_DIR / "analysis.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="TTS Audio generieren (Chatterbox)")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Nur bestimmte Personen verarbeiten")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bestehende TTS-Dateien ueberschreiben")
    args = parser.parse_args()

    analysis = load_analysis()
    TTS_DIR.mkdir(exist_ok=True)

    print("\n=== TTS Audio generieren (Chatterbox, lokal) ===\n")

    for entry in analysis:
        name = entry["name"]

        if args.only and name not in args.only:
            continue

        print(f"[{name}]")

        # Reference-Audio pruefen
        ref_path = None
        if entry.get("reference_audio"):
            ref_path = VIDEO_DIR / entry["reference_audio"]
        else:
            # Fallback: direkt im REF_DIR suchen
            candidate = REF_DIR / f"{name}_ref.wav"
            if candidate.exists():
                ref_path = candidate

        if not ref_path or not ref_path.exists():
            print(f"  Kein Reference-Audio. Zuerst python clone_voices.py ausfuehren.\n")
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

        out_path = TTS_DIR / f"{name}.wav"

        if out_path.exists() and not args.overwrite:
            print(f"  tts/{name}.wav existiert bereits (--overwrite zum Ueberschreiben)\n")
            continue

        print(f"  Reference: {ref_path.name}")
        print(f"  Text:      {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Generiere tts/{name}.wav...")

        try:
            generate_speech(text, ref_path, out_path)
            size_kb = out_path.stat().st_size // 1024
            print(f"  Gespeichert: tts/{name}.wav ({size_kb} KB)\n")
        except Exception as e:
            print(f"  Fehler: {e}\n")

    # Zusammenfassung
    tts_files = list(TTS_DIR.glob("*.wav")) + list(TTS_DIR.glob("*.mp3"))
    print(f"TTS fertig: {len(tts_files)} Datei(en) in tts/")
    for f in sorted(tts_files):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    print("\nNaechster Schritt:")
    print("  python -m lipsync.musetalk_batch  ->  Lipsync mit TTS-Audio")


if __name__ == "__main__":
    main()
