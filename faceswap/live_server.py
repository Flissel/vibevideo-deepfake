"""Live face-swap MJPEG server.

Spawned by the Automation_ui backend on demand. Reads eyeTerm's MJPEG
(http://127.0.0.1:8099/stream), runs each JPEG through InsightFace
inswapper, re-encodes JPEG, serves as multipart/x-mixed-replace on
http://127.0.0.1:8098/stream?target=Marshall.

Runs in voice/.venv312 because that's where the full ML stack
(insightface + onnxruntime-gpu) lives. The backend (.venv) cannot
import those, hence this is a separate process.

Usage:
    python -m faceswap.live_server [--port 8098]

Idle-shutdown: terminates self after 5 min without active clients to
free GPU/RAM. The backend respawns it on next /swap-stream request.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Make the package importable when run as a module
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))  # vibevideo_deepfake/

from faceswap.presets import resolve_preset, DISPLAY_NAMES  # noqa: E402
from faceswap.swapper import FaceSwapper                    # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("faceswap.live_server")

UPSTREAM = "http://127.0.0.1:8099/stream"
IDLE_TIMEOUT_S = 300  # kill self after 5 min idle

app = FastAPI(title="faceswap.live_server")

_swappers: dict = {}                       # slug -> FaceSwapper
_lock = threading.Lock()
_last_request_t = time.time()
_active_clients = 0
_active_clients_lock = threading.Lock()


def _norm_slug(target: str) -> str:
    """Accept display name or slug, return slug."""
    name_to_slug = {v.lower(): k for k, v in DISPLAY_NAMES.items()}
    if target.lower() in name_to_slug:
        return name_to_slug[target.lower()]
    return target


def _get_swapper(target: str) -> FaceSwapper:
    """Lazy-load + cache. Cold start ~12s, hot lookup <1ms."""
    slug = _norm_slug(target)
    with _lock:
        cached = _swappers.get(slug)
        if cached is not None:
            return cached
    target_path = resolve_preset(slug)
    logger.info("Loading FaceSwapper slug=%s file=%s", slug, target_path.name)
    sw = FaceSwapper(
        target_face_path=target_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    with _lock:
        _swappers[slug] = sw
    logger.info("FaceSwapper ready slug=%s", slug)
    return sw


def _iter_mjpeg_frames(resp):
    """Parse multipart/x-mixed-replace from an http.client.HTTPResponse."""
    boundary = b"--frame"
    buf = bytearray()
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            start = buf.find(boundary)
            if start < 0:
                break
            hdr_end = buf.find(b"\r\n\r\n", start)
            if hdr_end < 0:
                break
            header = bytes(buf[start:hdr_end]).decode("ascii", "ignore")
            length = 0
            for line in header.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    try:
                        length = int(line.split(":", 1)[1].strip())
                    except Exception:
                        length = 0
            if length <= 0:
                del buf[:hdr_end + 4]
                continue
            body_start = hdr_end + 4
            body_end = body_start + length
            if len(buf) < body_end:
                break
            jpeg = bytes(buf[body_start:body_end])
            del buf[:body_end + 2]
            yield jpeg


def _swap_jpeg(swapper: FaceSwapper, jpeg_bytes: bytes) -> bytes:
    """Decode JPEG → swap → encode JPEG. Pass-through original on failure."""
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg_bytes
        swapped, ist_face = swapper.swap(frame)
        out = swapped if ist_face is not None else frame
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return bytes(buf) if ok else jpeg_bytes
    except Exception as e:
        logger.warning("swap failed: %s", e)
        return jpeg_bytes


@app.get("/health")
async def health():
    with _lock:
        loaded = sorted(_swappers.keys())
    with _active_clients_lock:
        clients = _active_clients
    return {
        "ok": True,
        "loaded_targets": loaded,
        "active_clients": clients,
        "idle_s": round(time.time() - _last_request_t, 1),
    }


@app.get("/stream")
def stream(target: str = Query(...)):
    """Live swap-stream. Cold first-target ~12s, hot near-instant."""
    global _last_request_t
    _last_request_t = time.time()
    try:
        swapper = _get_swapper(target)
    except Exception as e:
        raise HTTPException(404, f"Cannot load target {target!r}: {e}")

    def gen():
        global _last_request_t, _active_clients
        with _active_clients_lock:
            _active_clients += 1
        try:
            resp = urlopen(UPSTREAM, timeout=10)
            for jpeg in _iter_mjpeg_frames(resp):
                _last_request_t = time.time()
                swapped = _swap_jpeg(swapper, jpeg)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(swapped)).encode() + b"\r\n\r\n"
                    + swapped + b"\r\n"
                )
        except Exception as e:
            logger.warning("stream error: %s", e)
        finally:
            with _active_clients_lock:
                _active_clients -= 1

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )


def _idle_watchdog():
    """Self-terminate after IDLE_TIMEOUT_S of no activity."""
    while True:
        time.sleep(30)
        with _active_clients_lock:
            clients = _active_clients
        idle = time.time() - _last_request_t
        if clients == 0 and idle > IDLE_TIMEOUT_S:
            logger.info("Idle for %.0fs, no clients — shutting down", idle)
            import os
            os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8098)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    threading.Thread(target=_idle_watchdog, daemon=True).start()
    logger.info("Live face-swap server on %s:%d (idle-kill after %ds)",
                args.host, args.port, IDLE_TIMEOUT_S)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
