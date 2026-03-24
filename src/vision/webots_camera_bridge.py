"""
VayuSwarm — Webots Camera Bridge
Receives JPEG camera frames from Webots drone controllers via TCP,
optionally sends them to Kaggle YOLO API, and pushes annotated frames
to the dashboard via WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import io
import socket
import struct
import threading
import time
from typing import Dict, Optional

import structlog
import yaml

logger = structlog.get_logger(__name__)

# ─── Per-drone frame store ─────────────────────────────────────────────────────
class DroneFrameStore:
    """Thread-safe store of the latest JPEG frame per drone."""

    def __init__(self):
        self._frames: Dict[str, bytes] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, drone_id: str, jpeg: bytes):
        with self._lock:
            self._frames[drone_id] = jpeg
            self._timestamps[drone_id] = time.time()

    def get(self, drone_id: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(drone_id)

    def get_all(self) -> Dict[str, bytes]:
        with self._lock:
            return dict(self._frames)

    def get_age_seconds(self, drone_id: str) -> float:
        with self._lock:
            t = self._timestamps.get(drone_id)
            return time.time() - t if t else float('inf')


_frame_store = DroneFrameStore()


# ─── TCP listener per drone ────────────────────────────────────────────────────
def _drone_camera_listener(drone_id: str, camera_port: int, remote_endpoint: Optional[str]):
    """Connects to a Webots drone camera stream and decodes frames."""
    log = logger.bind(drone_id=drone_id, port=camera_port)

    while True:
        try:
            log.info("camera_bridge.connecting")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(('localhost', camera_port))
            sock.settimeout(10.0)
            log.info("camera_bridge.connected")

            while True:
                # Read header: 32 bytes drone_id + 4 bytes frame length
                header = _recv_exactly(sock, 36)
                if not header:
                    break
                _did = header[:32].rstrip(b'\x00').decode()
                frame_len = struct.unpack('>I', header[32:36])[0]
                if frame_len == 0 or frame_len > 5_000_000:
                    continue
                jpeg = _recv_exactly(sock, frame_len)
                if not jpeg:
                    break

                annotated = jpeg
                if remote_endpoint:
                    annotated = _remote_yolo_annotate(jpeg, remote_endpoint) or jpeg

                _frame_store.update(drone_id, annotated)

        except Exception as e:
            log.warning("camera_bridge.error", error=str(e))
            time.sleep(3.0)
        finally:
            try:
                sock.close()
            except Exception:
                pass


def _recv_exactly(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _remote_yolo_annotate(jpeg: bytes, endpoint: str) -> Optional[bytes]:
    """Send frame to Kaggle YOLO API and draw boxes on the returned frame."""
    try:
        import requests
        import cv2
        import numpy as np

        resp = requests.post(
            endpoint,
            files={'file': ('frame.jpg', jpeg, 'image/jpeg')},
            timeout=2.0,
        )
        if resp.status_code != 200:
            return None

        detections = resp.json().get('detections', [])
        if not detections:
            return jpeg

        # Decode image to draw boxes
        nparr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        COLORS = {
            'person': (0, 255, 100),
            'car': (255, 165, 0), 'vehicle_car': (255, 165, 0),
            'truck': (255, 200, 0), 'vehicle_truck': (255, 200, 0),
            'motorcycle': (255, 220, 0), 'vehicle_motorcycle': (255, 220, 0),
        }
        DEFAULT_COLOR = (200, 200, 200)

        for d in detections:
            cls = d.get('class_name', 'unknown')
            conf = d.get('confidence', 0)
            x1, y1, x2, y2 = [int(v) for v in d.get('bbox', [0, 0, 0, 0])]
            color = COLORS.get(cls, DEFAULT_COLOR)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.0%}"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        _, out = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return out.tobytes()
    except Exception as e:
        logger.warning("yolo_annotate.error", error=str(e))
        return None


# ─── Public API ───────────────────────────────────────────────────────────────
class WebotsCameraBridge:
    """
    Connects to each Webots drone's camera TCP stream, optionally routes
    frames through Kaggle YOLO, and exposes get_frame() for the dashboard.
    """

    DRONE_CAMERA_PORTS = {
        'drone_01': 9001,
        'drone_02': 9002,
        'drone_03': 9003,
    }

    def __init__(self, config: dict):
        self._config = config
        vision = config.get('vision', {})
        yolo_cfg = vision.get('yolo', {})
        self._remote_endpoint: Optional[str] = yolo_cfg.get('remote_endpoint')
        self._threads: list[threading.Thread] = []

    def start(self):
        """Start listener threads for each drone camera."""
        for drone_id, port in self.DRONE_CAMERA_PORTS.items():
            t = threading.Thread(
                target=_drone_camera_listener,
                args=(drone_id, port, self._remote_endpoint),
                daemon=True,
                name=f"cam-bridge-{drone_id}",
            )
            t.start()
            self._threads.append(t)
        logger.info("webots_camera_bridge.started",
                    drones=list(self.DRONE_CAMERA_PORTS.keys()),
                    yolo_endpoint=self._remote_endpoint or "local")

    def get_frame_b64(self, drone_id: str) -> Optional[str]:
        """Return latest frame as base64 string (for WebSocket JSON)."""
        frame = _frame_store.get(drone_id)
        if frame is None:
            return None
        return base64.b64encode(frame).decode()

    def get_frame_bytes(self, drone_id: str) -> Optional[bytes]:
        return _frame_store.get(drone_id)

    def get_all_frames_b64(self) -> Dict[str, str]:
        result = {}
        for drone_id, frame in _frame_store.get_all().items():
            result[drone_id] = base64.b64encode(frame).decode()
        return result

    def is_alive(self, drone_id: str) -> bool:
        return _frame_store.get_age_seconds(drone_id) < 5.0
