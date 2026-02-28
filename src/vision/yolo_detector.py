"""
VayuSwarm — YOLOv8 Detection Module

Runs YOLOv8 inference on RGB camera frames.
Outputs structured DetectionEvent objects for:
  - person / vehicle detection (pre-trained)
  - weapon detection (rifle, handgun) — requires fine-tuned model
  - uniform classification (military, civilian) — requires fine-tuned model
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import structlog

from proto.messages import (
    BoundingBox,
    DetectionClass,
    DetectionEvent,
    DetectionSource,
    GeoPoint,
)
from src.vision.geo_tagger import GeoTagger

logger = structlog.get_logger(__name__)

# ─── COCO class ID → VayuSwarm class mapping ───────────────────────────────────
COCO_CLASS_MAP = {
    0: DetectionClass.PERSON,      # person
    1: DetectionClass.VEHICLE,     # bicycle (treat as vehicle)
    2: DetectionClass.VEHICLE,     # car
    3: DetectionClass.VEHICLE,     # motorcycle
    5: DetectionClass.VEHICLE,     # bus
    7: DetectionClass.VEHICLE,     # truck
    14: DetectionClass.ANIMAL,     # bird
    15: DetectionClass.ANIMAL,     # cat
    16: DetectionClass.ANIMAL,     # dog
    17: DetectionClass.ANIMAL,     # horse
    18: DetectionClass.ANIMAL,     # sheep
    19: DetectionClass.ANIMAL,     # cow
}

# Custom class map for VayuSwarm fine-tuned model (13 classes)
# Matches class_mapping.json from trained YOLOv8 on GitHub
CUSTOM_CLASS_MAP = {
    # Core detections
    "person": DetectionClass.PERSON,
    "vehicle_car": DetectionClass.VEHICLE_CAR,
    "vehicle_truck": DetectionClass.VEHICLE_TRUCK,
    "vehicle_motorcycle": DetectionClass.VEHICLE_MOTORCYCLE,
    # Weapons
    "weapon_rifle": DetectionClass.WEAPON_RIFLE,
    "weapon_pistol": DetectionClass.WEAPON_PISTOL,
    "weapon_knife": DetectionClass.WEAPON_KNIFE,
    # Uniforms → classify as PERSON, uniform type stored in metadata
    "uniform_military": DetectionClass.PERSON,
    "uniform_police": DetectionClass.PERSON,
    "uniform_civilian": DetectionClass.PERSON,
    # Special
    "drone": DetectionClass.DRONE,
    "suspicious_package": DetectionClass.SUSPICIOUS_PACKAGE,
    "fire": DetectionClass.FIRE,
    # VisDrone patrol model (5 classes — real aerial data)
    "car": DetectionClass.VEHICLE_CAR,
    "truck": DetectionClass.VEHICLE_TRUCK,
    "motorcycle": DetectionClass.VEHICLE_MOTORCYCLE,
    "bicycle": DetectionClass.VEHICLE,
}

# Uniform type extraction from YOLO class name
UNIFORM_CLASS_MAP = {
    "uniform_military": "military",
    "uniform_police": "police",
    "uniform_civilian": "civilian",
}


class YOLODetector:
    """
    YOLOv8 object detection wrapper.
    
    Loads pre-trained or fine-tuned YOLOv8 model and produces
    structured DetectionEvent objects per frame.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        class_thresholds: Optional[dict[str, float]] = None,
        hfov_deg: float = 90.0,
        vfov_deg: float = 60.0,
    ):
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._device = device
        self._class_thresholds = class_thresholds or {}
        self._model = None
        self._is_custom = False
        self._frame_count = 0
        self._total_detections = 0
        self._geo_tagger = GeoTagger(hfov_deg=hfov_deg, vfov_deg=vfov_deg)

    def load(self) -> None:
        """Load the YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)

            # Detect if this is a custom fine-tuned model
            model_names = getattr(self._model, "names", {})
            if model_names:
                # Detect custom model by checking if any name matches our custom class map
                custom_indicators = set(CUSTOM_CLASS_MAP.keys())
                model_name_set = set(model_names.values())
                if model_name_set & custom_indicators:
                    self._is_custom = True
                    logger.info("yolo.custom_model_detected", classes=list(model_names.values()))
                elif not model_name_set & {"person", "bicycle", "car", "airplane", "bus", "train", "truck", "boat"}:
                    # Doesn't look like COCO either — treat as custom
                    self._is_custom = True
                    logger.info("yolo.non_coco_model_detected", classes=list(model_names.values()))

            logger.info(
                "yolo.model_loaded",
                path=self._model_path,
                device=self._device,
                custom=self._is_custom,
            )
        except ImportError:
            logger.warning("yolo.ultralytics_not_installed, using mock mode")
            self._model = None
        except Exception as e:
            logger.error("yolo.load_error", error=str(e))
            self._model = None

    def detect(
        self,
        frame: np.ndarray,
        drone_position: Optional[GeoPoint] = None,
        heading_deg: float = 0.0,
    ) -> list[DetectionEvent]:
        """
        Run YOLOv8 inference on a single frame.
        
        Args:
            frame: BGR image (H, W, 3) numpy array
            drone_position: optional drone GPS for geo-referencing detections
            heading_deg: compass heading of drone (degrees, 0=North) for
                         rotating pixel offsets into geographic coordinates
            
        Returns:
            List of DetectionEvent objects with geo_position set to estimated
            ground GPS location of each detected object.
        """
        self._frame_count += 1

        if self._model is None:
            return self._mock_detect(frame, drone_position, heading_deg)

        try:
            results = self._model(
                frame,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                device=self._device if self._device != "auto" else None,
                verbose=False,
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                h, w = frame.shape[:2]

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()

                    # Map class
                    if self._is_custom:
                        cls_name = result.names.get(cls_id, "unknown")
                        det_class = CUSTOM_CLASS_MAP.get(cls_name, DetectionClass.UNKNOWN)
                    else:
                        det_class = COCO_CLASS_MAP.get(cls_id, DetectionClass.UNKNOWN)

                    # Skip unmapped classes
                    if det_class == DetectionClass.UNKNOWN:
                        continue

                    # Apply per-class confidence threshold
                    class_threshold = self._class_thresholds.get(det_class.value, self._conf_threshold)
                    if conf < class_threshold:
                        continue

                    # Normalized bounding box
                    bbox = BoundingBox(
                        x_min=float(xyxy[0] / w),
                        y_min=float(xyxy[1] / h),
                        x_max=float(xyxy[2] / w),
                        y_max=float(xyxy[3] / h),
                    )

                    # Build metadata
                    meta = {
                        "raw_class_id": cls_id,
                        "frame_number": self._frame_count,
                    }
                    if self._is_custom:
                        meta["raw_class_name"] = cls_name
                        # Store uniform type if this is a uniform detection
                        uniform_type = UNIFORM_CLASS_MAP.get(cls_name)
                        if uniform_type:
                            meta["uniform_type"] = uniform_type

                    detection = DetectionEvent(
                        source=DetectionSource.RGB,
                        detection_class=det_class,
                        confidence=conf,
                        bbox=bbox,
                        geo_position=drone_position,  # will be refined below
                        metadata=meta,
                    )
                    detections.append(detection)

            # Geo-tag: replace drone position with estimated object GPS
            if drone_position and detections:
                self._geo_tagger.tag_detections(detections, drone_position, heading_deg)

            self._total_detections += len(detections)

            if detections:
                logger.debug(
                    "yolo.detections",
                    count=len(detections),
                    classes=[d.detection_class.value for d in detections],
                    frame=self._frame_count,
                )

            return detections

        except Exception as e:
            logger.error("yolo.inference_error", error=str(e), frame=self._frame_count)
            return []

    def _mock_detect(
        self,
        frame: np.ndarray,
        drone_position: Optional[GeoPoint] = None,
        heading_deg: float = 0.0,
    ) -> list[DetectionEvent]:
        """Mock detection for testing without a real YOLO model."""
        import random

        detections = []
        # Simulate random detections (~20% chance per frame)
        if random.random() < 0.2:
            det_class = random.choice([
                DetectionClass.PERSON,
                DetectionClass.VEHICLE,
            ])
            conf = random.uniform(0.5, 0.95)
            cx, cy = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
            size = random.uniform(0.05, 0.15)

            detection = DetectionEvent(
                source=DetectionSource.RGB,
                detection_class=det_class,
                confidence=conf,
                bbox=BoundingBox(
                    x_min=max(0, cx - size),
                    y_min=max(0, cy - size),
                    x_max=min(1, cx + size),
                    y_max=min(1, cy + size),
                ),
                geo_position=drone_position,  # will be refined below
                metadata={"mock": True, "frame_number": self._frame_count},
            )
            detections.append(detection)

        # Geo-tag mock detections too
        if drone_position and detections:
            self._geo_tagger.tag_detections(detections, drone_position, heading_deg)

        return detections

    @property
    def stats(self) -> dict:
        """Get detector statistics."""
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._total_detections,
            "model_loaded": self._model is not None,
            "model_path": self._model_path,
            "is_custom": self._is_custom,
            "geo_tagger": self._geo_tagger.stats,
        }
