"""
export_transcripts.py - Transkripte als .txt Dateien exportieren
Erstellt eine editierbare .txt Datei pro Person im Ordner transcripts/

Dann:
  1. transcripts/Moritz.txt, Stefan.txt, Surya.txt oeffnen und Text anpassen
  2. python generate_tts.py  ->  generiert TTS-Audio mit geklonter Stimme
  3. python build_video.py   ->  Video mit neuer Stimme
"""

import json
from pathlib import Path

VIDEO_DIR   = Path(__file__).parent.parent
TRANS_DIR   = VIDEO_DIR / "transcripts"


def main():
    analysis_file = VIDEO_DIR / "analysis.json"
    with open(analysis_file, encoding="utf-8") as f:
        analysis = json.load(f)

    TRANS_DIR.mkdir(exist_ok=True)

    print("\nExportiere Transkripte nach transcripts/\n")

    for entry in analysis:
        name = entry["name"]
        text = entry.get("transcript", "").strip()
        out  = TRANS_DIR / f"{name}.txt"

        out.write_text(text, encoding="utf-8")

        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  {out.name}")
        print(f"    {preview}\n")

    print("Fertig! Oeffne die .txt Dateien, passe den Text an,")
    print("dann: python generate_tts.py")


if __name__ == "__main__":
    main()
