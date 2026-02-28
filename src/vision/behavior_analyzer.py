"""
VayuSwarm — Behavior Analyzer (Temporal Transformer)

Analyzes temporal sequences of FusedEvents to detect behavior patterns:
  - Movement speed & direction
  - Formation patterns
  - Evasion behavior
  - Proximity to no-go zones over time
  
Uses a lightweight Transformer architecture (<10M params) that processes
structured data, never raw pixels.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from proto.messages import (
    BehaviorType,
    FusedEvent,
    GeoPoint,
    ThreatLevel,
)

logger = structlog.get_logger(__name__)


class TrackHistory:
    """Maintains temporal history for a single tracked entity."""

    def __init__(self, track_id: str, max_history: int = 30):
        self.track_id = track_id
        self.events: deque[FusedEvent] = deque(maxlen=max_history)
        self.positions: deque[tuple[float, float, float]] = deque(maxlen=max_history)  # (lat, lon, time)
        self.speeds: deque[float] = deque(maxlen=max_history)
        self.headings: deque[float] = deque(maxlen=max_history)
        self.last_update: float = 0.0

    def add(self, event: FusedEvent) -> None:
        """Add a new event to the history."""
        self.events.append(event)
        if event.geo_position:
            self.positions.append((
                event.geo_position.lat,
                event.geo_position.lon,
                event.timestamp,
            ))
        self.last_update = event.timestamp

    def get_speed(self) -> float:
        """Compute average speed over recent history."""
        if len(self.positions) < 2:
            return 0.0
        speeds = []
        for i in range(1, len(self.positions)):
            p1 = self.positions[i - 1]
            p2 = self.positions[i]
            dt = p2[2] - p1[2]
            if dt > 0:
                dist = _geo_distance_fast(p1[0], p1[1], p2[0], p2[1])
                speeds.append(dist / dt)
        return float(np.mean(speeds)) if speeds else 0.0

    def get_heading(self) -> float:
        """Compute current heading in degrees."""
        if len(self.positions) < 2:
            return 0.0
        p1 = self.positions[-2]
        p2 = self.positions[-1]
        dlat = p2[0] - p1[0]
        dlon = p2[1] - p1[1]
        heading = math.degrees(math.atan2(dlon, dlat))
        return heading % 360

    def get_heading_variance(self) -> float:
        """High heading variance → erratic/evasive movement."""
        if len(self.positions) < 3:
            return 0.0
        headings = []
        for i in range(1, len(self.positions)):
            p1 = self.positions[i - 1]
            p2 = self.positions[i]
            dlat = p2[0] - p1[0]
            dlon = p2[1] - p1[1]
            h = math.degrees(math.atan2(dlon, dlat)) % 360
            headings.append(h)
        if len(headings) < 2:
            return 0.0
        # Circular variance
        sin_sum = sum(math.sin(math.radians(h)) for h in headings)
        cos_sum = sum(math.cos(math.radians(h)) for h in headings)
        n = len(headings)
        r = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / n
        return 1.0 - r  # 0 = constant direction, 1 = totally random


class BehaviorAnalyzer:
    """
    Temporal behavior analyzer using a sliding window of FusedEvents.
    
    The analyzer maintains per-entity track histories and computes
    behavior classifications based on movement patterns.
    
    Architecture:
    - Lightweight: Uses handcrafted features + optional Transformer
    - The Transformer receives structured features, never raw pixels
    - Features: [speed, heading, heading_variance, group_count,
                 distance_to_ngz, armed, time_in_area]
    """

    # Behavior model output class mapping (matches training script)
    BEHAVIOR_CLASSES = ["patrol", "evasive", "formation", "stationary", "approaching"]
    BEHAVIOR_CLASS_MAP = {
        "patrol": BehaviorType.PATROL,
        "evasive": BehaviorType.EVASIVE_MOVEMENT,
        "formation": BehaviorType.FORMATION,
        "stationary": BehaviorType.STATIONARY,
        "approaching": BehaviorType.APPROACHING,
    }

    def __init__(
        self,
        window_size: int = 30,
        model_path: Optional[str] = None,
        evasion_speed_threshold: float = 3.0,  # m/s
        evasion_heading_var_threshold: float = 0.4,
        formation_spacing_threshold: float = 5.0,  # meters
        stale_track_timeout_s: float = 30.0,
    ):
        self._window_size = window_size
        self._model_path = model_path
        self._evasion_speed = evasion_speed_threshold
        self._evasion_heading_var = evasion_heading_var_threshold
        self._formation_spacing = formation_spacing_threshold
        self._stale_timeout = stale_track_timeout_s
        self._tracks: dict[str, TrackHistory] = {}
        self._model = None
        self._model_type: Optional[str] = None
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._analysis_count = 0

    def load(self) -> None:
        """Load the Transformer behavior model (ONNX or PyTorch)."""
        if self._model_path:
            try:
                model_dir = Path(self._model_path).parent

                # Load normalization stats if available
                norm_mean_path = model_dir / "norm_mean.npy"
                norm_std_path = model_dir / "norm_std.npy"
                if norm_mean_path.exists() and norm_std_path.exists():
                    self._norm_mean = np.load(str(norm_mean_path))
                    self._norm_std = np.load(str(norm_std_path))
                    logger.info("behavior.normalization_loaded")

                if self._model_path.endswith(".onnx"):
                    import onnxruntime as ort
                    self._model = ort.InferenceSession(
                        self._model_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self._model_type = "onnx"
                    logger.info("behavior.onnx_model_loaded", path=self._model_path)
                else:
                    import torch
                    self._model = torch.load(
                        self._model_path, map_location="cpu", weights_only=True
                    )
                    self._model_type = "pytorch"
                    logger.info("behavior.pytorch_model_loaded", path=self._model_path)
            except Exception as e:
                logger.warning("behavior.model_load_failed", error=str(e))
                self._model = None
                self._model_type = None
        logger.info("behavior.analyzer_ready",
                     mode=self._model_type or "rule_based",
                     window=self._window_size)

    def analyze(self, fused_events: list[FusedEvent]) -> list[FusedEvent]:
        """
        Analyze a batch of FusedEvents, adding behavior classifications.
        
        This mutates the events in-place, setting their behavior
        and behavior_confidence fields.
        
        Args:
            fused_events: List of FusedEvents from SensorFusion
            
        Returns:
            The same list with behavior fields populated
        """
        self._analysis_count += 1
        now = time.time()

        for event in fused_events:
            track_id = self._get_track_id(event)
            track = self._get_or_create_track(track_id)
            track.add(event)

            if self._model is not None:
                behavior, confidence = self._ml_classify(track)
            else:
                behavior, confidence = self._rule_classify(track, event)

            event.behavior = behavior
            event.behavior_confidence = confidence
            event.movement_speed_ms = track.get_speed()
            event.movement_heading = track.get_heading()

            # Re-assess threat level considering behavior
            if behavior == BehaviorType.EVASIVE_MOVEMENT and event.threat_level.value < ThreatLevel.HIGH.value:
                event.threat_level = ThreatLevel.HIGH

        # Clean up stale tracks
        stale = [tid for tid, t in self._tracks.items() if now - t.last_update > self._stale_timeout]
        for tid in stale:
            del self._tracks[tid]

        return fused_events

    def _get_track_id(self, event: FusedEvent) -> str:
        """
        Generate a track ID for an event.
        In a full system, this would use a proper tracker (DeepSORT, etc.).
        For now, use event_id as a simple proxy.
        """
        # Simple proximity-based tracking
        if event.geo_position:
            for tid, track in self._tracks.items():
                if track.positions:
                    last_pos = track.positions[-1]
                    dist = _geo_distance_fast(
                        event.geo_position.lat, event.geo_position.lon,
                        last_pos[0], last_pos[1]
                    )
                    if dist < 20.0 and (time.time() - track.last_update) < 5.0:
                        return tid
        return event.event_id

    def _get_or_create_track(self, track_id: str) -> TrackHistory:
        """Get or create a track history for the given ID."""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackHistory(track_id, self._window_size)
        return self._tracks[track_id]

    def _rule_classify(self, track: TrackHistory, event: FusedEvent) -> tuple[BehaviorType, float]:
        """
        Rule-based behavior classification.
        
        Analyzes movement patterns from track history.
        """
        if len(track.events) < 2:
            return BehaviorType.UNKNOWN, 0.3

        speed = track.get_speed()
        heading_var = track.get_heading_variance()

        # ── Evasive Movement ──
        # High speed + frequent direction changes
        if speed > self._evasion_speed and heading_var > self._evasion_heading_var:
            confidence = min(0.95, 0.6 + heading_var * 0.3 + (speed / 10.0) * 0.2)
            return BehaviorType.EVASIVE_MOVEMENT, confidence

        # ── Erratic ──
        # Low speed but very high heading variance
        if heading_var > 0.6 and speed < self._evasion_speed:
            return BehaviorType.ERRATIC, 0.6

        # ── Formation ──
        # Multiple entities moving in organized pattern
        if event.group_count >= 3 and heading_var < 0.2:
            return BehaviorType.FORMATION, 0.7

        # ── Approaching ──
        # Moving toward the drone or a no-go zone
        if speed > 1.0 and event.in_ngz:
            return BehaviorType.APPROACHING, 0.7

        # ── Retreating ──
        # Moving away from the drone
        if speed > 2.0 and heading_var < 0.15:
            return BehaviorType.RETREATING, 0.5

        # ── Patrol ──
        # Moderate speed, relatively constant heading
        if 0.5 < speed < 3.0 and heading_var < 0.3:
            return BehaviorType.PATROL, 0.6

        # ── Stationary ──
        if speed < 0.5:
            return BehaviorType.STATIONARY, 0.8

        return BehaviorType.UNKNOWN, 0.3

    def _ml_classify(self, track: TrackHistory) -> tuple[BehaviorType, float]:
        """
        ML (Transformer) based behavior classification using ONNX Runtime.
        
        Input shape: [1, 30, 5] — 30-frame window with 5 features per frame
        Features: [x, y, speed, heading/360, acceleration]
        Output: 5 classes [patrol, evasive, formation, stationary, approaching]
        """
        features = self._extract_features(track)

        try:
            if self._model_type == "onnx":
                input_name = self._model.get_inputs()[0].name
                outputs = self._model.run(None, {input_name: features})
                logits = outputs[0][0]  # (num_classes,)

                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()

                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])

            elif self._model_type == "pytorch":
                import torch
                with torch.no_grad():
                    x = torch.tensor(features)
                    output = self._model(x)
                    probs = torch.softmax(output, dim=-1).squeeze()
                    class_idx = torch.argmax(probs).item()
                    confidence = probs[class_idx].item()
            else:
                return BehaviorType.UNKNOWN, 0.3

            # Map class index to BehaviorType
            if class_idx < len(self.BEHAVIOR_CLASSES):
                cls_name = self.BEHAVIOR_CLASSES[class_idx]
                return self.BEHAVIOR_CLASS_MAP.get(cls_name, BehaviorType.UNKNOWN), confidence
            else:
                return BehaviorType.UNKNOWN, 0.3

        except Exception as e:
            logger.error("behavior.ml_error", error=str(e))
            return BehaviorType.UNKNOWN, 0.3

    def _extract_features(self, track: TrackHistory) -> np.ndarray:
        """
        Extract structured feature array from track history.
        
        Matches training format: [batch=1, window=30, features=5]
        Features per timestep: [x, y, speed, heading/360, acceleration]
        """
        n_features = 5
        seq = np.zeros((self._window_size, n_features), dtype=np.float32)

        positions = list(track.positions)
        events = list(track.events)

        for i, idx in enumerate(range(max(0, len(positions) - self._window_size), len(positions))):
            if i >= self._window_size:
                break

            pos = positions[idx]
            x, y, t = pos[0], pos[1], pos[2]

            # Compute speed and acceleration
            speed = 0.0
            accel = 0.0
            heading = 0.0
            if idx > 0:
                prev = positions[idx - 1]
                dt = t - prev[2]
                if dt > 0:
                    dist = _geo_distance_fast(prev[0], prev[1], x, y)
                    speed = dist / dt
                    heading = math.degrees(math.atan2(y - prev[1], x - prev[0])) % 360
                    if idx > 1:
                        prev2 = positions[idx - 2]
                        dt2 = prev[2] - prev2[2]
                        if dt2 > 0:
                            prev_speed = _geo_distance_fast(prev2[0], prev2[1], prev[0], prev[1]) / dt2
                            accel = (speed - prev_speed) / dt

            seq[i] = [x, y, speed, heading / 360.0, accel]

        # Apply normalization if available
        if self._norm_mean is not None and self._norm_std is not None:
            seq = (seq - self._norm_mean) / self._norm_std

        return seq[np.newaxis, ...].astype(np.float32)  # (1, 30, 5)

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    @property
    def stats(self) -> dict:
        return {
            "analysis_count": self._analysis_count,
            "active_tracks": self.active_tracks,
            "model_loaded": self._model is not None,
        }


def _geo_distance_fast(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fast approximate distance in meters (equirectangular projection)."""
    R = 6371000
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return R * math.sqrt(x * x + y * y)
