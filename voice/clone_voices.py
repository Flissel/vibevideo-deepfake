"""
clone_voices.py - ElevenLabs Voice Cloning
Extrahiert Audio aus jedem Video und erstellt Voice-Klone in ElevenLabs.
Speichert die Voice-IDs in analysis.json.

Verwendung:
  python clone_voices.py --api-key YOUR_KEY
  python clone_voices.py  (liest ELEVENLABS_API_KEY aus .env oder Umgebungsvariable)

Ergebnis in analysis.json pro Person:
  "voice_id": "abc123..."   <- ElevenLabs Voice ID
  "voice_name": "Moritz"    <- Name des Klons in ElevenLabs
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent
AUDIO_SAMPLE_DURATION = None   # None = ganzes Video, oder z.B. 60 fuer 60s


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


def extract_audio(video_path: Path, out_path: Path, duration: float = None):
    """Extrahiert Audio aus Video als MP3 fuer ElevenLabs Upload."""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-y", "-i", str(video_path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += [
        "-ac", "1",        # Mono
        "-ar", "44100",    # 44.1kHz (ElevenLabs Standard)
        "-b:a", "128k",    # 128kbps
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def clone_voice(client, name: str, audio_path: Path) -> str:
    """Laed Audio hoch und erstellt einen Instant Voice Clone. Gibt Voice-ID zurueck."""
    print(f"  Lade Audio hoch fuer '{name}'...")
    with open(audio_path, "rb") as f:
        voice = client.voices.ivc.create(
            name=name,
            description=f"Team member voice clone: {name}",
            files=[("audio.mp3", f, "audio/mpeg")],
        )
    return voice.voice_id


def get_existing_voices(client) -> dict:
    """Gibt alle vorhandenen Voices als {name: voice_id} zurueck."""
    voices = client.voices.get_all()
    return {v.name: v.voice_id for v in voices.voices}


def main():
    parser = argparse.ArgumentParser(description="ElevenLabs Voice Cloning")
    parser.add_argument("--api-key", type=str, help="ElevenLabs API Key")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Nur bestimmte Personen klonen")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bestehende Voice-Klone ueberschreiben (Standard: ueberspringen)")
    args = parser.parse_args()

    # API Key ermitteln
    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        # .env Datei versuchen
        env_file = VIDEO_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ELEVENLABS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"')
                    break

    if not api_key:
        print("Fehler: Kein API Key gefunden.")
        print("  Option 1: python clone_voices.py --api-key YOUR_KEY")
        print("  Option 2: ELEVENLABS_API_KEY=YOUR_KEY in .env Datei")
        sys.exit(1)

    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)

    # Existierende Voices abfragen
    print("Verbinde mit ElevenLabs...")
    try:
        existing = get_existing_voices(client)
        print(f"  {len(existing)} bestehende Voice(s) im Account gefunden.")
    except Exception as e:
        print(f"Verbindungsfehler: {e}")
        sys.exit(1)

    # Analysis laden
    analysis = load_analysis()
    changed = False

    for entry in analysis:
        name = entry["name"]

        if args.only and name not in args.only:
            continue

        print(f"\n[{name}]")

        if not entry.get("file"):
            print(f"  Kein Video in analysis.json, wird uebersprungen.")
            continue

        video_path = VIDEO_DIR / entry["file"]

        # Schon geklont?
        if not args.overwrite and entry.get("voice_id"):
            print(f"  Bereits geklont: {entry['voice_id']} (--overwrite zum Ueberschreiben)")
            continue

        # Name schon in ElevenLabs?
        if not args.overwrite and name in existing:
            print(f"  Voice '{name}' bereits in ElevenLabs: {existing[name]}")
            entry["voice_id"] = existing[name]
            entry["voice_name"] = name
            changed = True
            continue

        if not video_path.exists():
            print(f"  WARNUNG: Video nicht gefunden: {video_path.name}")
            continue

        # Audio extrahieren
        tmp_audio = VIDEO_DIR / f"_tmp_voice_{name}.mp3"
        try:
            print(f"  Extrahiere Audio aus {video_path.name}...")
            extract_audio(video_path, tmp_audio, duration=AUDIO_SAMPLE_DURATION)

            file_size_mb = tmp_audio.stat().st_size / 1024 / 1024
            print(f"  Audio: {file_size_mb:.1f} MB")

            # Voice klonen
            voice_id = clone_voice(client, name, tmp_audio)
            print(f"  Voice-ID: {voice_id}")

            entry["voice_id"] = voice_id
            entry["voice_name"] = name
            changed = True

        except Exception as e:
            print(f"  Fehler beim Klonen: {e}")
        finally:
            if tmp_audio.exists():
                tmp_audio.unlink()

    # Speichern
    if changed:
        save_analysis(analysis)
        print(f"\nVoice-IDs in analysis.json gespeichert.")

    # Zusammenfassung
    print("\n=== Zusammenfassung ===")
    for entry in analysis:
        vid = entry.get("voice_id", "(nicht geklont)")
        print(f"  {entry['name']:<12} Voice-ID: {vid}")


if __name__ == "__main__":
    main()
