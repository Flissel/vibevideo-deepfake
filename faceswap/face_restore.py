"""Face restoration — GFPGAN post-processing for the inswapper output.

The inswapper core runs at 128x128; its swapped face is upscaled to
frame resolution and ends up soft, noisy, low on skin detail. A
restoration model re-synthesises pores, sharpness and fine texture,
which is the single biggest visible step toward deepfake-grade output.

GFPGANv1.4 (ONNX, ~340 MB) is the default. It expects a 512x512 face,
RGB, normalised to [-1, 1], NCHW; it returns the restored face in the
same layout.

The model file ships with VisoMaster's asset bundle:
    E:/Vibemind_Tools/VisoMaster/model_assets/GFPGANv1.4.onnx
Override with FACE_RESTORE_MODEL if it lives elsewhere.

Restoration is applied only to the face crop and feather-pasted back,
so the rest of the frame is untouched and the crop edge has no seam.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get(
    "FACE_RESTORE_MODEL",
    "E:/Vibemind_Tools/VisoMaster/model_assets/GFPGANv1.4.onnx",
)


class FaceRestorer:
    """GFPGAN ONNX wrapper — lazy, GPU-first."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        import onnxruntime as ort

        path = model_path or _DEFAULT_MODEL
        if not Path(path).exists():
            raise FileNotFoundError(
                f"face-restore model not found: {path}. Set FACE_RESTORE_MODEL "
                "or install VisoMaster's model_assets."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(path, providers=providers)
        self._inp = self._sess.get_inputs()[0].name
        self._size = 512
        logger.info("FaceRestorer ready — %s", Path(path).name)

    def _restore_512(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run GFPGAN on a face crop; returns same-size restored crop."""
        h, w = face_bgr.shape[:2]
        face = cv2.resize(face_bgr, (self._size, self._size))
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5            # → [-1, 1]
        blob = rgb.transpose(2, 0, 1)[None]         # NCHW
        out = self._sess.run(None, {self._inp: blob})[0][0]
        out = (out.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h))

    def restore_face_region(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        blend: float = 1.0,
        feather_px: int = 24,
    ) -> np.ndarray:
        """Restore the face inside bbox and feather-paste it back.

        Args:
            frame_bgr: full swapped frame
            bbox: (x0,y0,x1,y1) of the face — typically the insight Face.bbox
            blend: 0..1 — how strongly the restored crop replaces the raw
                one (1.0 = full restore, lower keeps some original texture)
            feather_px: Gaussian-feather radius on the crop edge so the
                restored region has no hard boundary against the softer
                surrounding frame.

        Returns a new frame; the input is not modified.
        """
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox
        # pad the bbox a bit — GFPGAN works better with some margin
        bw, bh = x1 - x0, y1 - y0
        pad_x, pad_y = bw * 0.25, bh * 0.25
        cx0 = max(0, int(x0 - pad_x))
        cy0 = max(0, int(y0 - pad_y))
        cx1 = min(w, int(x1 + pad_x))
        cy1 = min(h, int(y1 + pad_y))
        if cx1 - cx0 < 20 or cy1 - cy0 < 20:
            return frame_bgr

        crop = frame_bgr[cy0:cy1, cx0:cx1]
        restored = self._restore_512(crop)
        if blend < 1.0:
            restored = cv2.addWeighted(restored, blend, crop, 1.0 - blend, 0)

        # Feather mask: 1 in the crop interior, fading to 0 at the edge.
        ch, cw = crop.shape[:2]
        m = np.ones((ch, cw), dtype=np.float32)
        k = feather_px
        if k > 0 and ch > 2 * k and cw > 2 * k:
            m[:k, :] = 0; m[-k:, :] = 0
            m[:, :k] = 0; m[:, -k:] = 0
            ksize = k * 2 + 1
            m = cv2.GaussianBlur(m, (ksize, ksize), k / 2.0)
        m3 = m[..., None]

        out = frame_bgr.copy()
        out[cy0:cy1, cx0:cx1] = (
            restored.astype(np.float32) * m3
            + crop.astype(np.float32) * (1.0 - m3)
        ).clip(0, 255).astype(np.uint8)
        return out
