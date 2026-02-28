"""
VayuSwarm — Sensor Fusion Engine

Merges RGB (YOLO) + Thermal detections into unified FusedEvent objects.
The Transformer/LLM never sees raw pixels — only structured fused data.
"""

from __future__ import annotations

import time
from typing import Optional

import structlog

from proto.messages import (
    BoundingBox,
    DetectionClass,
    DetectionEvent,
    DetectionSource,
    FusedEvent,
    GeoPoint,
    ThermalDetection,
    ThreatLevel,
    UniformType,
)

logger = structlog.get_logger(__name__)


class SensorFusion:
    """
    Fuses RGB and Thermal detections into unified FusedEvent objects.
    
    Pipeline:
    1. Spatial matching (IoU between YOLO boxes and thermal blob positions)
    2. Confidence boosting (both sources agree → higher confidence)
    3. Threat assessment (armed + uniform + proximity → threat level)
    4. Produces FusedEvent with all available information
    """

    def __init__(
        self,
        iou_match_threshold: float = 0.3,
        confidence_boost: float = 0.15,
        no_go_zones: Optional[list] = None,
    ):
        self._iou_threshold = iou_match_threshold
        self._confidence_boost = confidence_boost
        self._no_go_zones = no_go_zones or []
        self._fusion_count = 0

    def fuse(
        self,
        rgb_detections: list[DetectionEvent],
        thermal_detections: list[ThermalDetection],
        drone_position: Optional[GeoPoint] = None,
    ) -> list[FusedEvent]:
        """
        Merge RGB and thermal detections into FusedEvents.
        
        Strategy:
        - Match RGB and thermal detections by spatial proximity
        - Unmatched RGB detections become FusedEvents with RGB-only sources
        - Unmatched thermal detections become FusedEvents with thermal-only sources
        - Matched pairs get confidence boost and combined data
        """
        self._fusion_count += 1
        fused_events = []

        # Track which detections have been matched
        matched_rgb = set()
        matched_thermal = set()

        # Step 1: Try to match RGB → Thermal
        for i, rgb_det in enumerate(rgb_detections):
            best_match_idx = -1
            best_match_score = 0.0

            for j, therm_det in enumerate(thermal_detections):
                if j in matched_thermal:
                    continue

                score = self._match_score(rgb_det, therm_det)
                if score > self._iou_threshold and score > best_match_score:
                    best_match_score = score
                    best_match_idx = j

            if best_match_idx >= 0:
                # Matched pair — merge
                fused = self._merge_matched(
                    rgb_det, thermal_detections[best_match_idx], drone_position
                )
                fused_events.append(fused)
                matched_rgb.add(i)
                matched_thermal.add(best_match_idx)
            else:
                # Unmatched RGB
                fused = self._from_rgb_only(rgb_det, drone_position)
                fused_events.append(fused)
                matched_rgb.add(i)

        # Step 2: Unmatched thermal detections
        for j, therm_det in enumerate(thermal_detections):
            if j not in matched_thermal:
                fused = self._from_thermal_only(therm_det, drone_position)
                fused_events.append(fused)

        # Step 3: Compute threat levels
        for event in fused_events:
            event.threat_level = self._assess_threat(event)
            event.in_ngz = self._check_ngz(event.geo_position)

        if fused_events:
            logger.debug(
                "fusion.result",
                total=len(fused_events),
                rgb_count=len(rgb_detections),
                thermal_count=len(thermal_detections),
                matched_pairs=len(matched_rgb & {i for i, _ in enumerate(rgb_detections) if i in matched_rgb}),
            )

        return fused_events

    def _match_score(
        self,
        rgb_det: DetectionEvent,
        therm_det: ThermalDetection,
    ) -> float:
        """
        Compute match score between an RGB detection and a thermal detection.
        Uses class compatibility + spatial proximity.
        """
        score = 0.0

        # Class compatibility
        if rgb_det.detection_class == therm_det.blob_class:
            score += 0.5
        elif (rgb_det.detection_class in (DetectionClass.PERSON, DetectionClass.UNKNOWN)
              and therm_det.blob_class in (DetectionClass.PERSON, DetectionClass.UNKNOWN)):
            score += 0.3

        # Spatial proximity (if both have geo positions)
        if rgb_det.geo_position and therm_det.geo_position:
            dist = self._geo_distance(rgb_det.geo_position, therm_det.geo_position)
            if dist < 10.0:  # Within 10 meters
                score += 0.5 * (1.0 - dist / 10.0)

        # If no geo, use confidence as weak signal
        if not rgb_det.geo_position and not therm_det.geo_position:
            score += 0.2  # Assume potential match in simulation

        return score

    def _merge_matched(
        self,
        rgb_det: DetectionEvent,
        therm_det: ThermalDetection,
        drone_position: Optional[GeoPoint],
    ) -> FusedEvent:
        """Merge a matched RGB + Thermal detection pair."""
        # Use RGB class but boost confidence
        combined_conf = min(1.0, (rgb_det.confidence + therm_det.confidence) / 2.0 + self._confidence_boost)

        # Check if armed (weapon detected)
        armed = rgb_det.detection_class in (DetectionClass.WEAPON_RIFLE, DetectionClass.WEAPON_HANDGUN)
        weapon_class = rgb_det.detection_class if armed else None

        # Primary class — if weapon detected, the person carrying it is the entity
        primary_class = DetectionClass.PERSON if armed else rgb_det.detection_class

        return FusedEvent(
            detection_class=primary_class,
            class_confidence=combined_conf,
            armed=armed,
            armed_confidence=rgb_det.confidence if armed else 0.0,
            weapon_class=weapon_class,
            geo_position=rgb_det.geo_position or therm_det.geo_position or drone_position,
            bbox=rgb_det.bbox,
            temperature_c=therm_det.temperature_c,
            group_count=therm_det.group_count,
            sources=[DetectionSource.RGB, DetectionSource.THERMAL],
        )

    def _from_rgb_only(
        self,
        rgb_det: DetectionEvent,
        drone_position: Optional[GeoPoint],
    ) -> FusedEvent:
        """Create FusedEvent from RGB detection only."""
        armed = rgb_det.detection_class in (DetectionClass.WEAPON_RIFLE, DetectionClass.WEAPON_HANDGUN)
        primary_class = DetectionClass.PERSON if armed else rgb_det.detection_class

        return FusedEvent(
            detection_class=primary_class,
            class_confidence=rgb_det.confidence,
            armed=armed,
            armed_confidence=rgb_det.confidence if armed else 0.0,
            weapon_class=rgb_det.detection_class if armed else None,
            geo_position=rgb_det.geo_position or drone_position,
            bbox=rgb_det.bbox,
            sources=[DetectionSource.RGB],
        )

    def _from_thermal_only(
        self,
        therm_det: ThermalDetection,
        drone_position: Optional[GeoPoint],
    ) -> FusedEvent:
        """Create FusedEvent from thermal detection only."""
        return FusedEvent(
            detection_class=therm_det.blob_class,
            class_confidence=therm_det.confidence * 0.8,  # Lower confidence for thermal-only
            geo_position=therm_det.geo_position or drone_position,
            temperature_c=therm_det.temperature_c,
            group_count=therm_det.group_count,
            sources=[DetectionSource.THERMAL],
        )

    def _assess_threat(self, event: FusedEvent) -> ThreatLevel:
        """
        Compute threat level based on all available signals.
        """
        score = 0.0

        # Armed is a strong threat signal
        if event.armed:
            score += 3.0 * event.armed_confidence

        # Person in no-go zone
        if event.in_ngz and event.detection_class == DetectionClass.PERSON:
            score += 2.0

        # Military uniform
        if event.uniform == UniformType.MILITARY:
            score += 1.0

        # Group size
        if event.group_count > 3:
            score += 0.5

        # Evasive behavior (will be set by BehaviorAnalyzer later)
        # behavior scoring happens after fusion

        # High confidence detection
        if event.class_confidence > 0.85:
            score += 0.3

        # Multi-source confirmation
        if len(event.sources) > 1:
            score += 0.3

        # Map score to threat level
        if score >= 4.0:
            return ThreatLevel.CRITICAL
        elif score >= 2.5:
            return ThreatLevel.HIGH
        elif score >= 1.5:
            return ThreatLevel.MEDIUM
        elif score >= 0.5:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _check_ngz(self, position: Optional[GeoPoint]) -> bool:
        """Check if a position is inside any no-go zone."""
        if not position or not self._no_go_zones:
            return False

        for zone in self._no_go_zones:
            if hasattr(zone, "is_no_go") and zone.is_no_go:
                if self._point_in_polygon(position, zone.points):
                    return True
        return False

    @staticmethod
    def _point_in_polygon(point: GeoPoint, polygon: list[GeoPoint]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            if ((polygon[i].lat > point.lat) != (polygon[j].lat > point.lat) and
                    point.lon < (polygon[j].lon - polygon[i].lon) *
                    (point.lat - polygon[i].lat) /
                    (polygon[j].lat - polygon[i].lat) + polygon[i].lon):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _geo_distance(p1: GeoPoint, p2: GeoPoint) -> float:
        """Approximate distance in meters between two GPS points (Haversine)."""
        import math
        R = 6371000  # Earth radius in meters
        dlat = math.radians(p2.lat - p1.lat)
        dlon = math.radians(p2.lon - p1.lon)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(p1.lat)) * math.cos(math.radians(p2.lat)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def set_no_go_zones(self, zones: list) -> None:
        """Update no-go zones for threat assessment."""
        self._no_go_zones = [z for z in zones if getattr(z, "is_no_go", False)]
        logger.info("fusion.ngz_updated", count=len(self._no_go_zones))

    @property
    def stats(self) -> dict:
        return {"fusion_count": self._fusion_count}
