"""
VayuSwarm — Thermal Signature Analysis Module

Processes thermal camera frames to:
  - Classify heat blobs (person vs animal vs vehicle)
  - Estimate group count and spacing
  - Output ThermalDetection events
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import structlog

from proto.messages import (
    DetectionClass,
    GeoPoint,
    ThermalDetection,
)

logger = structlog.get_logger(__name__)


class ThermalModel:
    """
    Thermal signature analyzer.
    
    Uses a combination of temperature thresholding and a lightweight CNN
    classifier to identify heat sources in thermal frames.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        person_temp_min_c: float = 30.0,
        person_temp_max_c: float = 42.0,
        vehicle_temp_min_c: float = 40.0,
        vehicle_temp_max_c: float = 120.0,
        animal_temp_min_c: float = 28.0,
        animal_temp_max_c: float = 40.0,
        min_blob_area: float = 0.001,    # Min relative area to consider
        max_blob_area: float = 0.5,       # Max relative area
    ):
        self._model_path = model_path
        self._person_temp = (person_temp_min_c, person_temp_max_c)
        self._vehicle_temp = (vehicle_temp_min_c, vehicle_temp_max_c)
        self._animal_temp = (animal_temp_min_c, animal_temp_max_c)
        self._min_blob_area = min_blob_area
        self._max_blob_area = max_blob_area
        self._model = None
        self._frame_count = 0

    # Thermal model output class mapping (matches training script)
    THERMAL_CLASSES = ["background", "human", "vehicle", "animal", "fire"]
    THERMAL_CLASS_MAP = {
        "background": (DetectionClass.UNKNOWN, False),
        "human": (DetectionClass.PERSON, True),
        "vehicle": (DetectionClass.VEHICLE, True),
        "animal": (DetectionClass.ANIMAL, True),
        "fire": (DetectionClass.FIRE, True),
    }

    def load(self) -> None:
        """Load the thermal classification model (ONNX or PyTorch)."""
        if self._model_path:
            try:
                if self._model_path.endswith(".onnx"):
                    import onnxruntime as ort
                    self._model = ort.InferenceSession(
                        self._model_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self._model_type = "onnx"
                    logger.info("thermal.onnx_model_loaded", path=self._model_path)
                else:
                    import torch
                    from torchvision import models as tv_models
                    import torch.nn as nn
                    # Reconstruct MobileNetV3 Small architecture
                    model = tv_models.mobilenet_v3_small(weights=None)
                    model.classifier[-1] = nn.Linear(
                        model.classifier[-1].in_features, len(self.THERMAL_CLASSES)
                    )
                    model.load_state_dict(
                        torch.load(self._model_path, map_location="cpu", weights_only=True)
                    )
                    model.eval()
                    self._model = model
                    self._model_type = "pytorch"
                    logger.info("thermal.pytorch_model_loaded", path=self._model_path)
            except Exception as e:
                logger.warning("thermal.model_load_failed", error=str(e))
                self._model = None
                self._model_type = None
        else:
            logger.info("thermal.using_threshold_mode (no ML model)")

    def detect(
        self,
        thermal_frame: np.ndarray,
        drone_position: Optional[GeoPoint] = None,
    ) -> list[ThermalDetection]:
        """
        Analyze a thermal frame and extract heat signature detections.
        
        Args:
            thermal_frame: Thermal image (H, W) or (H, W, 1) in temperature values
                          or normalized 0-255 grayscale
            drone_position: Optional drone GPS for geo-referencing
            
        Returns:
            List of ThermalDetection events
        """
        self._frame_count += 1

        if thermal_frame is None or thermal_frame.size == 0:
            return []

        # Ensure 2D
        if len(thermal_frame.shape) == 3:
            thermal_frame = thermal_frame[:, :, 0]

        h, w = thermal_frame.shape

        # Normalize to temperature-like values if needed
        # If frame is 0-255 uint8, map to approximate temperature range
        if thermal_frame.dtype == np.uint8:
            temp_frame = thermal_frame.astype(np.float32) / 255.0 * 50.0 + 10.0  # Map to 10-60°C
        else:
            temp_frame = thermal_frame.astype(np.float32)

        detections = []

        try:
            # Step 1: Threshold to find warm blobs
            warm_mask = temp_frame > self._person_temp[0]

            if not np.any(warm_mask):
                return []

            # Step 2: Find connected components (blobs)
            blobs = self._find_blobs(warm_mask)

            for blob in blobs:
                area_ratio = blob["area"] / (h * w)

                if area_ratio < self._min_blob_area or area_ratio > self._max_blob_area:
                    continue

                # Get average temperature of blob
                avg_temp = np.mean(temp_frame[blob["mask"]])

                # Classify based on temperature and size
                det_class, confidence = self._classify_blob(avg_temp, area_ratio, blob)

                detection = ThermalDetection(
                    blob_class=det_class,
                    confidence=confidence,
                    temperature_c=float(avg_temp),
                    blob_area=float(area_ratio),
                    geo_position=drone_position,
                    group_count=blob.get("sub_count", 1),
                    group_spacing_m=blob.get("spacing_m", 0.0),
                )
                detections.append(detection)

        except Exception as e:
            logger.error("thermal.detect_error", error=str(e), frame=self._frame_count)

        if detections:
            logger.debug("thermal.detections", count=len(detections), frame=self._frame_count)

        return detections

    def _find_blobs(self, mask: np.ndarray) -> list[dict]:
        """
        Find connected components (blobs) in a binary mask.
        Uses cv2 if available, falls back to simple implementation.
        """
        blobs = []

        try:
            import cv2
            mask_uint8 = mask.astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)

            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                cx, cy = centroids[i]
                blob_mask = labels == i

                # Estimate sub-entities within a large blob
                sub_count = max(1, area // 500)  # Rough estimate

                blobs.append({
                    "label": i,
                    "area": area,
                    "centroid": (cx, cy),
                    "mask": blob_mask,
                    "bbox": (
                        stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT],
                    ),
                    "sub_count": sub_count,
                    "spacing_m": 2.0 * sub_count,  # Rough spacing estimate
                })
        except ImportError:
            # Fallback: treat entire warm region as one blob
            area = int(np.sum(mask))
            if area > 0:
                ys, xs = np.where(mask)
                blobs.append({
                    "label": 1,
                    "area": area,
                    "centroid": (float(np.mean(xs)), float(np.mean(ys))),
                    "mask": mask,
                    "bbox": (int(xs.min()), int(ys.min()),
                             int(xs.max() - xs.min()), int(ys.max() - ys.min())),
                    "sub_count": 1,
                    "spacing_m": 0.0,
                })

        return blobs

    def _classify_blob(
        self,
        avg_temp: float,
        area_ratio: float,
        blob: dict,
    ) -> tuple[DetectionClass, float]:
        """
        Classify a thermal blob based on temperature and physical characteristics.
        
        Returns:
            (DetectionClass, confidence)
        """
        # If ML model is available, use it
        if self._model is not None:
            return self._ml_classify(blob)

        # Rule-based classification
        confidence = 0.5

        # Person: body temperature range, moderate size
        if self._person_temp[0] <= avg_temp <= self._person_temp[1]:
            if 0.001 < area_ratio < 0.05:
                confidence = 0.7 + min(0.2, (avg_temp - 33.0) / 10.0)
                return DetectionClass.PERSON, min(confidence, 0.95)

        # Vehicle: higher temperature, larger size
        if avg_temp > self._vehicle_temp[0]:
            if area_ratio > 0.02:
                confidence = 0.6 + min(0.2, (avg_temp - 50.0) / 70.0)
                return DetectionClass.VEHICLE, min(confidence, 0.90)

        # Animal: lower body temperature, small-medium size
        if self._animal_temp[0] <= avg_temp <= self._animal_temp[1]:
            if area_ratio < 0.02:
                return DetectionClass.ANIMAL, 0.5

        # Default: unknown
        return DetectionClass.UNKNOWN, 0.3

    def _ml_classify(self, blob: dict) -> tuple[DetectionClass, float]:
        """ML-based blob classification using trained ONNX or PyTorch model."""
        try:
            # Extract and preprocess blob region
            blob_mask = blob["mask"]
            bbox = blob.get("bbox")  # (x, y, w, h)

            if bbox is None:
                return DetectionClass.PERSON, 0.6

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                return DetectionClass.UNKNOWN, 0.3

            # Crop blob region from mask (approximate thermal patch)
            # Use the blob's bounding box region
            y_end = min(blob_mask.shape[0], y + h)
            x_end = min(blob_mask.shape[1], x + w)
            y_start = max(0, y)
            x_start = max(0, x)

            # Create a grayscale patch from the blob mask (0/255)
            patch = (blob_mask[y_start:y_end, x_start:x_end].astype(np.float32) * 255.0)

            if patch.size == 0:
                return DetectionClass.UNKNOWN, 0.3

            # Resize to model input size (224x224)
            try:
                import cv2
                patch_resized = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                # Simple nearest-neighbor resize fallback
                h_out, w_out = 224, 224
                h_in, w_in = patch.shape
                rows = (np.arange(h_out) * h_in // h_out).astype(int)
                cols = (np.arange(w_out) * w_in // w_out).astype(int)
                patch_resized = patch[np.ix_(rows, cols)]

            # Normalize to [0, 1] then apply mean/std [0.5, 0.5, 0.5]
            patch_norm = patch_resized / 255.0
            patch_norm = (patch_norm - 0.5) / 0.5

            # Convert grayscale to 3-channel (matches training: Lambda(x.repeat(3,1,1)))
            patch_3ch = np.stack([patch_norm, patch_norm, patch_norm], axis=0)  # (3, 224, 224)
            input_tensor = patch_3ch[np.newaxis, ...].astype(np.float32)  # (1, 3, 224, 224)

            if getattr(self, "_model_type", None) == "onnx":
                # ONNX Runtime inference
                input_name = self._model.get_inputs()[0].name
                outputs = self._model.run(None, {input_name: input_tensor})
                probs = outputs[0][0]  # (num_classes,)

                # Softmax
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()

                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])

            elif getattr(self, "_model_type", None) == "pytorch":
                import torch
                with torch.no_grad():
                    x = torch.tensor(input_tensor)
                    output = self._model(x)
                    probs = torch.softmax(output, dim=-1).squeeze()
                    class_idx = torch.argmax(probs).item()
                    confidence = probs[class_idx].item()
            else:
                return DetectionClass.PERSON, 0.6

            # Map class index to DetectionClass
            if class_idx < len(self.THERMAL_CLASSES):
                cls_name = self.THERMAL_CLASSES[class_idx]
                det_class, valid = self.THERMAL_CLASS_MAP.get(
                    cls_name, (DetectionClass.UNKNOWN, False)
                )
                if not valid:
                    return DetectionClass.UNKNOWN, confidence
                return det_class, confidence
            else:
                return DetectionClass.UNKNOWN, 0.3

        except Exception as e:
            logger.error("thermal.ml_classify_error", error=str(e))
            return DetectionClass.PERSON, 0.6

    def mock_detect(
        self,
        drone_position: Optional[GeoPoint] = None,
    ) -> list[ThermalDetection]:
        """Generate mock thermal detections for simulation/testing."""
        import random

        self._frame_count += 1
        detections = []

        if random.random() < 0.15:
            det_class = random.choice([DetectionClass.PERSON, DetectionClass.VEHICLE])
            temp = random.uniform(33.0, 39.0) if det_class == DetectionClass.PERSON else random.uniform(50.0, 80.0)

            detection = ThermalDetection(
                blob_class=det_class,
                confidence=random.uniform(0.5, 0.9),
                temperature_c=temp,
                blob_area=random.uniform(0.005, 0.05),
                geo_position=drone_position,
                group_count=random.randint(1, 3),
                group_spacing_m=random.uniform(1.0, 5.0),
            )
            detections.append(detection)

        return detections

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "model_loaded": self._model is not None,
        }
