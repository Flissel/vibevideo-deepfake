"""
setup_wav2lip.py - Wav2Lip Setup Script
Laedt Wav2Lip herunter und installiert alle Abhaengigkeiten.

Einmalig ausfuehren:
  python setup_wav2lip.py

Danach lip_sync.py verwenden.
"""

import subprocess
import sys
import os
from pathlib import Path

VIDEO_DIR   = Path(__file__).parent.parent
WAV2LIP_DIR = VIDEO_DIR / "wav2lip"
MODEL_URL   = "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
DETECTOR_URL = "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/s3fd-619a316812.pth"


def run(cmd, **kwargs):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"  FEHLER (code {result.returncode})")
        sys.exit(1)


def main():
    pip = str(VIDEO_DIR / "venv/Scripts/pip.exe")
    python = str(VIDEO_DIR / "venv/Scripts/python.exe")

    print("\n=== Wav2Lip Setup ===\n")

    # 1. Repo klonen
    if not WAV2LIP_DIR.exists():
        print("1. Klone Wav2Lip Repo...")
        run(["git", "clone", "https://github.com/justinjohn0306/Wav2Lip.git",
             str(WAV2LIP_DIR)])
    else:
        print("1. Wav2Lip Repo bereits vorhanden.")

    # 2. Abhaengigkeiten installieren
    print("\n2. Installiere Abhaengigkeiten...")
    packages = [
        "opencv-python",
        "librosa",
        "basicsr",
        "facexlib",
        "realesrgan",
        "batch-face",
    ]
    run([pip, "install"] + packages)

    # 3. Modell-Ordner erstellen
    models_dir = WAV2LIP_DIR / "checkpoints"
    models_dir.mkdir(exist_ok=True)

    face_det_dir = WAV2LIP_DIR / "face_detection" / "detection" / "sfd"
    face_det_dir.mkdir(parents=True, exist_ok=True)

    # 4. Modelle herunterladen
    import urllib.request

    model_path = models_dir / "wav2lip_gan.pth"
    if not model_path.exists():
        print(f"\n3. Lade Wav2Lip GAN Modell (~420 MB)...")
        print(f"   -> {model_path}")
        urllib.request.urlretrieve(MODEL_URL, str(model_path),
            reporthook=lambda b, bs, t: print(f"\r   {min(b*bs, t)/1024/1024:.0f}/{t/1024/1024:.0f} MB", end=""))
        print()
    else:
        print(f"\n3. Wav2Lip Modell bereits vorhanden: {model_path.name}")

    detector_path = face_det_dir / "s3fd-619a316812.pth"
    if not detector_path.exists():
        print(f"\n4. Lade Face Detector (~85 MB)...")
        urllib.request.urlretrieve(DETECTOR_URL, str(detector_path),
            reporthook=lambda b, bs, t: print(f"\r   {min(b*bs, t)/1024/1024:.0f}/{t/1024/1024:.0f} MB", end=""))
        print()
    else:
        print(f"4. Face Detector bereits vorhanden.")

    print("\n=== Setup abgeschlossen! ===")
    print("\nNaechster Schritt:")
    print("  python lip_sync.py    ->  Lip Sync fuer alle Videos")


if __name__ == "__main__":
    main()
