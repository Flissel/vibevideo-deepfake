"""
clone_voices.py - Reference-Audio Extraktion fuer Chatterbox Voice Cloning
Extrahiert 10-15s sauberes Audio aus jedem Video als Reference fuer Chatterbox.
Speichert Reference-Pfade in analysis.json.

Chatterbox braucht kein separates Cloning/API — das Reference-Audio wird
direkt beim TTS-Generate mitgegeben.

Verwendung:
  python clone_voices.py
  python clone_voices.py --only Moritz
  python clone_voices.py --overwrite

Ergebnis in analysis.json pro Person:
  "reference_audio": "voice_references/Moritz_ref.wav"
"""

import argparse
import json
import subprocess
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent
REF_DIR = Path.home() / ".rowboat" / "Videos" / "voice_references"
REFERENCE_DURATION = 15  # Sekunden Reference-Audio


def get_ffmpeg_path() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_analysis() -> list:
    analysis_file = VIDEO_DIR / "analysis.json"
    if not analysis_file.exists():
        raise FileNotFoundError("analysis.json nicht gefunden. Zuerst 'python analyze.py' ausfuehren.")
    with open(analysis_file, encoding="utf-8") as f:
        return json.load(f)


def save_analysis(data: list):
    analysis_file = VIDEO_DIR / "analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_reference_audio(video_path: Path, out_path: Path, duration: float = REFERENCE_DURATION):
    """Extrahiert sauberes Reference-Audio (WAV, mono, 44.1kHz) fuer Chatterbox."""
    ffmpeg = get_ffmpeg_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-t", str(duration),
        "-ac", "1",        # Mono
        "-ar", "44100",    # 44.1kHz
        "-acodec", "pcm_s16le",  # WAV PCM
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    parser = argparse.ArgumentParser(description="Reference-Audio Extraktion fuer Chatterbox")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Nur bestimmte Personen verarbeiten")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bestehende Reference-Dateien ueberschreiben")
    parser.add_argument("--duration", type=float, default=REFERENCE_DURATION,
                        help=f"Dauer des Reference-Clips in Sekunden (Standard: {REFERENCE_DURATION})")
    args = parser.parse_args()

    analysis = load_analysis()
    changed = False

    REF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Reference-Audio Extraktion (Chatterbox, lokal) ===\n")

    for entry in analysis:
        name = entry["name"]

        if args.only and name not in args.only:
            continue

        print(f"[{name}]")

        if not entry.get("file"):
            print(f"  Kein Video in analysis.json, wird uebersprungen.")
            continue

        video_path = VIDEO_DIR / entry["file"]
        ref_path = REF_DIR / f"{name}_ref.wav"

        # Schon extrahiert?
        if not args.overwrite and ref_path.exists():
            print(f"  Reference existiert: {ref_path.name} (--overwrite zum Ueberschreiben)")
            if "reference_audio" not in entry:
                entry["reference_audio"] = str(ref_path.relative_to(VIDEO_DIR))
                changed = True
            continue

        if not video_path.exists():
            print(f"  WARNUNG: Video nicht gefunden: {video_path.name}")
            continue

        # Reference-Audio extrahieren
        try:
            print(f"  Extrahiere {args.duration}s Reference aus {video_path.name}...")
            extract_reference_audio(video_path, ref_path, duration=args.duration)
            size_kb = ref_path.stat().st_size // 1024
            print(f"  Gespeichert: {ref_path.name} ({size_kb} KB)")

            entry["reference_audio"] = str(ref_path.relative_to(VIDEO_DIR))
            # Legacy-Felder entfernen
            entry.pop("voice_id", None)
            entry.pop("voice_name", None)
            changed = True

        except Exception as e:
            print(f"  Fehler: {e}")

    if changed:
        save_analysis(analysis)
        print(f"\nReference-Pfade in analysis.json gespeichert.")

    # Zusammenfassung
    print("\n=== Zusammenfassung ===")
    for entry in analysis:
        ref = entry.get("reference_audio", "(nicht extrahiert)")
        print(f"  {entry['name']:<12} Reference: {ref}")

    print("\nKein API-Key noetig! Naechster Schritt:")
    print("  python generate_tts.py  ->  TTS mit Chatterbox generieren")


if __name__ == "__main__":
    main()
