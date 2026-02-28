"""
VayuSwarm — Camera Source Module

Provides RGB + thermal frame capture from:
  - Video file (for replay/testing)
  - USB/CSI camera device
  - RTSP stream
  - Synthetic (random noise for simulation)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CameraMode(str, Enum):
    SYNTHETIC = "synthetic"
    VIDEO_FILE = "video_file"
    DEVICE = "device"
    RTSP = "rtsp"


class CameraSource:
    """
    Unified camera source supporting video files, device capture, and synthetic frames.
    
    Usage:
        cam = CameraSource(source="path/to/video.mp4")  # Video file
        cam = CameraSource(source=0)                     # USB camera device 0
        cam = CameraSource(source="rtsp://...")           # RTSP stream
        cam = CameraSource()                              # Synthetic random frames
        
        cam.open()
        frame = cam.read()  # Returns (H, W, 3) BGR numpy array
        cam.release()
    """

    def __init__(
        self,
        source: Optional[str | int] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        loop: bool = True,
    ):
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._loop = loop
        self._cap = None
        self._frame_count = 0
        self._mode = self._detect_mode(source)

    def _detect_mode(self, source) -> CameraMode:
        """Detect what kind of source this is."""
        if source is None:
            return CameraMode.SYNTHETIC
        if isinstance(source, int):
            return CameraMode.DEVICE
        if isinstance(source, str):
            if source.startswith("rtsp://") or source.startswith("http://"):
                return CameraMode.RTSP
            if Path(source).exists() or Path(source).suffix in (".mp4", ".avi", ".mkv", ".mov", ".jpg", ".png"):
                return CameraMode.VIDEO_FILE
            # Try as device index
            try:
                int(source)
                return CameraMode.DEVICE
            except ValueError:
                pass
        return CameraMode.SYNTHETIC

    @property
    def mode(self) -> CameraMode:
        return self._mode

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        if self._mode == CameraMode.SYNTHETIC:
            return True
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        """Open the camera source. Returns True on success."""
        if self._mode == CameraMode.SYNTHETIC:
            logger.info("camera.synthetic_mode", width=self._width, height=self._height)
            return True

        try:
            import cv2

            if self._mode == CameraMode.DEVICE:
                dev = int(self._source) if isinstance(self._source, str) else self._source
                self._cap = cv2.VideoCapture(dev)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            elif self._mode in (CameraMode.VIDEO_FILE, CameraMode.RTSP):
                self._cap = cv2.VideoCapture(str(self._source))
            else:
                return False

            if not self._cap.isOpened():
                logger.error("camera.open_failed", source=str(self._source))
                self._cap = None
                return False

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                "camera.opened",
                mode=self._mode.value,
                source=str(self._source),
                resolution=f"{actual_w}x{actual_h}",
                fps=actual_fps,
                total_frames=total_frames if total_frames > 0 else "live",
            )
            return True

        except ImportError:
            logger.warning("camera.opencv_not_available, falling back to synthetic")
            self._mode = CameraMode.SYNTHETIC
            return True
        except Exception as e:
            logger.error("camera.open_error", error=str(e))
            return False

    def read(self) -> Optional[np.ndarray]:
        """
        Read the next frame.
        
        Returns:
            BGR frame (H, W, 3) numpy array, or None if no frame available.
        """
        self._frame_count += 1

        if self._mode == CameraMode.SYNTHETIC:
            return self._synthetic_frame()

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()

        if not ret:
            if self._loop and self._mode == CameraMode.VIDEO_FILE:
                # Loop back to beginning
                self._cap.set(2, 0)  # cv2.CAP_PROP_POS_FRAMES = 1, but safer to use 2 = POS_MSEC
                import cv2
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
                logger.debug("camera.video_looped", frame=self._frame_count)
            else:
                return None

        return frame

    def read_thermal(self) -> Optional[np.ndarray]:
        """
        Read a synthetic thermal frame.
        
        In a real system, this would come from a separate thermal camera.
        For now, generates synthetic thermal data.
        """
        return np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    def _synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic RGB frame (random noise)."""
        return np.random.randint(0, 255, (self._height, self._width, 3), dtype=np.uint8)

    def release(self) -> None:
        """Release the camera source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("camera.released", mode=self._mode.value, frames=self._frame_count)

    @property
    def stats(self) -> dict:
        return {
            "mode": self._mode.value,
            "source": str(self._source) if self._source else "synthetic",
            "frames_read": self._frame_count,
            "is_open": self.is_open,
        }

    def __del__(self):
        try:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        except Exception:
            pass
