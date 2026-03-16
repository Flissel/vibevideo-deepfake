"""
lip_sync.py - Lip Sync mit Wav2Lip + Soft Mask Blending

Synchronisiert Lippenbewegungen. Wav2Lip-Output wird per Soft Mask
zurueck auf Original geblendet → kein sichtbarer Mund-Box-Artefakt.

Verwendung:
  python lip_sync.py                          # alle, TTS Audio
  python lip_sync.py --original-audio         # Original-Ton (kein ElevenLabs)
  python lip_sync.py --only Moritz            # nur eine Person
  python lip_sync.py --overwrite              # bestehende ueberschreiben
  python lip_sync.py --no-blend               # Wav2Lip-Output ohne Soft Mask

Ergebnis: lip_sync/<Name>.mp4
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR    = Path(__file__).parent.parent
WAV2LIP_DIR  = VIDEO_DIR / "wav2lip"
TTS_DIR      = VIDEO_DIR / "tts"
OUT_DIR      = VIDEO_DIR / "lip_sync"
MODEL_PATH   = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
GFPGAN_PATH  = WAV2LIP_DIR / "checkpoints" / "GFPGANv1.4.pth"

# Temporal smoothing buffer size (frames) — higher = smoother mask, less flicker
TEMPORAL_BUFFER = 7


def get_ffmpeg_path() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_analysis() -> list:
    with open(VIDEO_DIR / "analysis.json", encoding="utf-8") as f:
        return json.load(f)


def extract_clip(entry: dict, out_path: Path):
    """Schneidet das relevante Segment aus dem Original-Video."""
    ffmpeg = get_ffmpeg_path()
    src = VIDEO_DIR / entry["file"]
    start = entry.get("clip_start", 0.0)
    end   = entry.get("clip_end", entry["duration"])
    duration = end - start

    subprocess.run([
        ffmpeg, "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(src),
        "-c:v", "libx264", "-c:a", "aac",
        str(out_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def extract_audio(video_path: Path, out_path: Path):
    """Extrahiert Audio aus einem Video als WAV."""
    ffmpeg = get_ffmpeg_path()
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(out_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def run_wav2lip(video_path: Path, audio_path: Path, out_path: Path):
    """Fuehrt Wav2Lip aus."""
    script = WAV2LIP_DIR / "inference.py"

    cmd = [
        sys.executable, str(script),
        "--checkpoint_path", str(MODEL_PATH),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(out_path),
        "--resize_factor", "1",
        "--pads", "0", "15", "0", "0",
    ]
    print(f"  Wav2Lip laeuft (kann 5-15 Min dauern auf CPU)...")
    result = subprocess.run(cmd, cwd=str(WAV2LIP_DIR))
    if result.returncode != 0:
        raise RuntimeError("Wav2Lip fehlgeschlagen.")


FACE_MODEL_PATH = VIDEO_DIR / "face_landmarker.task"


# ---------------------------------------------------------------------------
# CodeFormer Face Restoration (sharper than GFPGAN)
# ---------------------------------------------------------------------------

_codeformer_restorer = None

def get_codeformer():
    """Lazy-init CodeFormer restorer (Transformer-based, sharper than GFPGAN)."""
    global _codeformer_restorer
    if _codeformer_restorer is None:
        from codeformer.app import inference_app
        # This initializes the global models in codeformer.app
        # We import the function and use it directly
        _codeformer_restorer = True
        print("  CodeFormer geladen")
    return _codeformer_restorer


def sharpen_unsharp(img_bgr: np.ndarray, strength: float = 0.5, sigma: float = 2.0) -> np.ndarray:
    """Unsharp mask sharpening — crisp lips without ringing."""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def restore_face(frame_bgr: np.ndarray,
                  codeformer_fidelity: float = 0.5,
                  unsharp_strength: float = 0.5,
                  unsharp_sigma: float = 2.0) -> np.ndarray:
    """CodeFormer face restoration + bilateral filter + sharpening."""
    from codeformer.app import inference_app
    import tempfile

    h, w = frame_bgr.shape[:2]

    # CodeFormer expects file path — use temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, frame_bgr)

    try:
        output = inference_app(
            image=tmp_path,
            background_enhance=False,
            face_upsample=False,
            upscale=1,
            codeformer_fidelity=codeformer_fidelity,
        )
    finally:
        import os
        os.unlink(tmp_path)

    # output is RGB numpy array from CodeFormer
    if output is not None:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return frame_bgr  # fallback

    # Bilateral filter: denoise while preserving edges
    output = cv2.bilateralFilter(output, d=5, sigmaColor=50, sigmaSpace=50)
    # Unsharp mask for crisp lip detail
    output = sharpen_unsharp(output, strength=unsharp_strength, sigma=unsharp_sigma)
    return output


# ---------------------------------------------------------------------------
# Color Histogram Matching (LAB space)
# ---------------------------------------------------------------------------

def match_color_lab(source_bgr: np.ndarray, target_bgr: np.ndarray,
                    mask: np.ndarray = None) -> np.ndarray:
    """
    Match color/brightness of source to target using LAB mean/std transfer.
    If mask is provided (float32, 0-1), only statistics from masked region are used.
    Returns color-corrected source_bgr.
    """
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is not None and mask.sum() > 100:
        # Use mask to compute stats only in mouth region
        mask_bool = mask > 0.3
        for ch in range(3):
            src_ch = src_lab[:, :, ch]
            tgt_ch = tgt_lab[:, :, ch]
            src_mean = src_ch[mask_bool].mean()
            src_std  = max(src_ch[mask_bool].std(), 1e-6)
            tgt_mean = tgt_ch[mask_bool].mean()
            tgt_std  = max(tgt_ch[mask_bool].std(), 1e-6)
            src_lab[:, :, ch] = (src_ch - src_mean) * (tgt_std / src_std) + tgt_mean
    else:
        for ch in range(3):
            src_mean = src_lab[:, :, ch].mean()
            src_std  = max(src_lab[:, :, ch].std(), 1e-6)
            tgt_mean = tgt_lab[:, :, ch].mean()
            tgt_std  = max(tgt_lab[:, :, ch].std(), 1e-6)
            src_lab[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (tgt_std / src_std) + tgt_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

# Mouth outer contour landmark indices in MediaPipe Face Mesh (478 landmarks)
MOUTH_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]


def create_face_landmarker():
    """Erstellt einen FaceLandmarker mit dem neuen MediaPipe Tasks API."""
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def create_mouth_mask(frame_rgb: np.ndarray, landmarker,
                      expand_px: int = None,
                      blur_radius: int = None,
                      expand_ratio: float = 0.04,
                      blur_ratio: float = 0.05) -> np.ndarray | None:
    """
    Erstellt eine adaptive weiche Maske um den Mundbereich.
    expand_px und blur_radius werden proportional zur Gesichtsgroesse berechnet,
    wenn nicht explizit gesetzt.
    Gibt float32 Array (H, W) mit Werten 0-1 zurueck, oder None wenn kein Gesicht.
    """
    import mediapipe as mp

    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]

    # Adaptive sizing: compute face height from forehead (10) to chin (152)
    face_top_y = lms[10].y * h
    face_bot_y = lms[152].y * h
    face_height = max(face_bot_y - face_top_y, 50)

    if expand_px is None:
        expand_px = max(5, int(face_height * expand_ratio))
    if blur_radius is None:
        blur_radius = max(5, int(face_height * blur_ratio))

    pts = np.array(
        [[int(lms[i].x * w), int(lms[i].y * h)] for i in MOUTH_OUTER],
        dtype=np.int32
    )

    # Include chin: minimal extension for natural transition
    chin_extend = int(face_height * 0.03)
    chin_pts = pts.copy()
    # Push bottom points down slightly for chin coverage
    mouth_center_y = pts[:, 1].mean()
    for i in range(len(chin_pts)):
        if chin_pts[i, 1] > mouth_center_y:
            chin_pts[i, 1] += chin_extend

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(chin_pts), 255)

    kernel_size = max(3, expand_px * 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    ksize = blur_radius * 2 + 1
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                               (ksize, ksize), blur_radius)
    return mask_f


def blend_videos(original_path: Path, synced_path: Path,
                 out_path: Path, audio_path: Path,
                 use_gfpgan: bool = True,
                 lipsync_params: dict = None):
    """
    Blendet Wav2Lip-Output per Soft Mask auf Originalvideo.
    Pipeline: Wav2Lip -> GFPGAN restore -> Color match -> Adaptive mask -> Temporal smooth -> Blend
    Nur der Mundbereich wird ersetzt — Rest bleibt 100% original.
    """
    # Per-person params with defaults
    p = lipsync_params or {}
    p_temporal_buffer = p.get("temporal_buffer", TEMPORAL_BUFFER)
    p_codeformer_fidelity = p.get("codeformer_fidelity", 0.5)
    p_unsharp_strength = p.get("unsharp_strength", 0.5)
    p_unsharp_sigma = p.get("unsharp_sigma", 2.0)
    p_expand_ratio = p.get("expand_ratio", 0.04)
    p_blur_ratio = p.get("blur_ratio", 0.05)

    features = ["Adaptive Mask", "Temporal Smooth"]
    if use_gfpgan:
        features.insert(0, "CodeFormer")
    features.append("Color Match")
    print(f"  Enhanced Blending: {' + '.join(features)}")
    if lipsync_params:
        print(f"  Per-person params: temporal_buffer={p_temporal_buffer}, "
              f"fidelity={p_codeformer_fidelity}, blur_ratio={p_blur_ratio}")

    cap_orig  = cv2.VideoCapture(str(original_path))
    cap_synced = cv2.VideoCapture(str(synced_path))

    fps    = cap_orig.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temp-Video ohne Audio
    tmp_video = out_path.with_suffix(".tmp_blend.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (w, h))

    landmarker = create_face_landmarker()

    frame_idx = 0
    last_mask = None
    mask_buffer = []  # Temporal smoothing buffer

    while True:
        ret_o, orig_bgr   = cap_orig.read()
        ret_s, synced_bgr = cap_synced.read()
        if not ret_o or not ret_s:
            break

        # Groesse angleichen falls unterschiedlich
        if synced_bgr.shape[:2] != (h, w):
            synced_bgr = cv2.resize(synced_bgr, (w, h))

        # Step 1: CodeFormer face restoration on Wav2Lip output
        if use_gfpgan:
            try:
                synced_bgr = restore_face(synced_bgr,
                                          codeformer_fidelity=p_codeformer_fidelity,
                                          unsharp_strength=p_unsharp_strength,
                                          unsharp_sigma=p_unsharp_sigma)
                if synced_bgr.shape[:2] != (h, w):
                    synced_bgr = cv2.resize(synced_bgr, (w, h))
            except Exception:
                pass  # Fallback: use unrestored frame

        # Step 2: Adaptive mouth mask from original frame
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        mask = create_mouth_mask(orig_rgb, landmarker,
                                expand_ratio=p_expand_ratio,
                                blur_ratio=p_blur_ratio)

        if mask is not None:
            last_mask = mask
        elif last_mask is not None:
            mask = last_mask
        else:
            writer.write(synced_bgr)
            frame_idx += 1
            continue

        # Step 3: Temporal smoothing — average mask over last N frames
        mask_buffer.append(mask)
        if len(mask_buffer) > p_temporal_buffer:
            mask_buffer.pop(0)
        smoothed_mask = np.mean(mask_buffer, axis=0).astype(np.float32)

        # Step 4: Color matching in mouth region (LAB space) — single pass
        synced_bgr = match_color_lab(synced_bgr, orig_bgr, smoothed_mask)

        # Step 5: Soft mask blend — original * (1-mask) + enhanced * mask
        mask_3ch = np.stack([smoothed_mask, smoothed_mask, smoothed_mask], axis=2)
        orig_f   = orig_bgr.astype(np.float32)
        synced_f = synced_bgr.astype(np.float32)
        blended  = (orig_f * (1.0 - mask_3ch) + synced_f * mask_3ch).astype(np.uint8)

        writer.write(blended)
        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == n_orig:
            print(f"\r  Blending: {frame_idx}/{n_orig} ({frame_idx/max(n_orig,1)*100:.0f}%)  ",
                  end="", flush=True)

    print()
    cap_orig.release()
    cap_synced.release()
    writer.release()
    landmarker.close()

    # Audio einmixen — frame-count-anchored sync
    from sync_guard import count_frames, verify_sync
    n_frames = count_frames(tmp_video)
    auth_duration = n_frames / fps
    print(f"  Sync: {n_frames} frames @ {fps:.1f}fps -> {auth_duration:.3f}s")

    ffmpeg = get_ffmpeg_path()
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(tmp_video),
        "-i", str(audio_path),
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-t", f"{auth_duration:.6f}",
        "-map", "0:v:0", "-map", "1:a:0",
        str(out_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    tmp_video.unlink(missing_ok=True)

    # Verify sync
    vr = verify_sync(out_path)
    status = "OK" if vr["is_synced"] else f"DRIFT {vr['delta_ms']:.1f}ms"
    print(f"  Blend fertig: {out_path.name} (sync: {status})")


def main():
    parser = argparse.ArgumentParser(description="Lip Sync mit Wav2Lip + Soft Mask")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--original-audio", action="store_true",
                        help="Original-Audio verwenden statt TTS (kein ElevenLabs noetig)")
    parser.add_argument("--no-blend", action="store_true",
                        help="Soft Mask Blending ueberspringen")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("Wav2Lip Modell nicht gefunden.")
        print("Bitte zuerst ausfuehren: python setup_wav2lip.py")
        sys.exit(1)

    analysis = load_analysis()
    OUT_DIR.mkdir(exist_ok=True)

    mode = "Original-Audio" if args.original_audio else "TTS"
    blend = "an" if not args.no_blend else "aus"
    print(f"\nLip Sync mit Wav2Lip  |  Audio: {mode}  |  Soft Mask: {blend}\n")

    for entry in analysis:
        name = entry["name"]

        if args.only and name not in args.only:
            continue

        if entry.get("skip_lipsync"):
            print(f"[{name}] skip_lipsync=true, uebersprungen.\n")
            continue

        if not entry.get("file") or not (VIDEO_DIR / entry["file"]).exists():
            print(f"[{name}] Kein Video, uebersprungen.\n")
            continue

        print(f"[{name}]")

        out_path = OUT_DIR / f"{name}.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"  lip_sync/{name}.mp4 existiert (--overwrite zum Ueberschreiben)\n")
            continue

        tmp_clip = VIDEO_DIR / f"_tmp_lipsync_{name}.mp4"
        tmp_audio = VIDEO_DIR / f"_tmp_lipsync_{name}_audio.wav"
        tmp_wav2lip = VIDEO_DIR / f"_tmp_lipsync_{name}_w2l.mp4"

        try:
            # 1. Clip extrahieren
            start = entry.get("clip_start", 0.0)
            end   = entry.get("clip_end", entry["duration"])
            print(f"  Extrahiere Clip {start:.1f}s - {end:.1f}s...")
            extract_clip(entry, tmp_clip)

            # 2. Audio bestimmen
            if args.original_audio:
                audio_path = tmp_audio
                print(f"  Extrahiere Original-Audio...")
                extract_audio(tmp_clip, audio_path)
            else:
                audio_path = TTS_DIR / f"{name}.mp3"
                if not audio_path.exists():
                    print(f"  Kein TTS: tts/{name}.mp3 — nutze Original-Audio als Fallback")
                    audio_path = tmp_audio
                    extract_audio(tmp_clip, audio_path)

            # 3. Wav2Lip
            if args.no_blend:
                run_wav2lip(tmp_clip, audio_path, out_path)
            else:
                run_wav2lip(tmp_clip, audio_path, tmp_wav2lip)

                # 4. Soft Mask Blend (with per-person params if available)
                blend_videos(tmp_clip, tmp_wav2lip, out_path, audio_path,
                             lipsync_params=entry.get("lipsync_params"))

            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  Gespeichert: lip_sync/{name}.mp4 ({size_mb:.1f} MB)\n")

        except Exception as e:
            import traceback
            print(f"  Fehler: {e}")
            traceback.print_exc()
            print()
        finally:
            for tmp in [tmp_clip, tmp_audio, tmp_wav2lip]:
                if tmp.exists():
                    tmp.unlink()

    synced = list(OUT_DIR.glob("*.mp4"))
    print(f"Fertig: {len(synced)} lip-gesynced Video(s) in lip_sync/")
    for f in synced:
        print(f"  {f.name}")

    if synced:
        print("\nNaechster Schritt:")
        print("  python split_screen.py --use-synced")


if __name__ == "__main__":
    main()
