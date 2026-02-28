"""
Tests for src/vision/geo_tagger.py — GPS geo-tagging from pixel bounding boxes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from proto.messages import BoundingBox, DetectionClass, DetectionEvent, DetectionSource, GeoPoint
from src.vision.geo_tagger import GeoTagger, _METRES_PER_DEG_LAT


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tagger() -> GeoTagger:
    """90° HFOV × 60° VFOV nadir camera."""
    return GeoTagger(hfov_deg=90.0, vfov_deg=60.0)


@pytest.fixture
def drone_pos() -> GeoPoint:
    """Drone hovering at 100 m AGL over a convenient test location."""
    return GeoPoint(lat=28.6139, lon=77.2090, alt=100.0)


@pytest.fixture
def center_bbox() -> BoundingBox:
    """Bounding box centred exactly in the middle of the frame."""
    return BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Centre pixel → should be ≈ directly below the drone
# ──────────────────────────────────────────────────────────────────────────────

class TestCentrePixel:
    def test_centre_maps_to_drone_position(self, tagger, drone_pos, center_bbox):
        result = tagger.pixel_to_gps(center_bbox, drone_pos)
        dist = tagger.haversine_distance(
            GeoPoint(lat=drone_pos.lat, lon=drone_pos.lon, alt=0.0),
            result,
        )
        # Centre should be within 1 m of directly below the drone
        assert dist < 1.0, f"Centre-pixel offset too large: {dist:.2f} m"

    def test_result_object_altitude_is_zero(self, tagger, drone_pos, center_bbox):
        result = tagger.pixel_to_gps(center_bbox, drone_pos)
        assert result.alt == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Directional offsets (nadir camera, heading=0 = North)
# ──────────────────────────────────────────────────────────────────────────────

class TestDirectionalOffsets:
    def test_right_half_is_east(self, tagger, drone_pos):
        """Object on right side of image → east of drone (lon increases)."""
        right_bbox = BoundingBox(x_min=0.7, y_min=0.4, x_max=0.9, y_max=0.6)
        result = tagger.pixel_to_gps(right_bbox, drone_pos, heading_deg=0.0)
        assert result.lon > drone_pos.lon, "Right-half object should be east"

    def test_left_half_is_west(self, tagger, drone_pos):
        """Object on left side → west of drone (lon decreases)."""
        left_bbox = BoundingBox(x_min=0.1, y_min=0.4, x_max=0.3, y_max=0.6)
        result = tagger.pixel_to_gps(left_bbox, drone_pos, heading_deg=0.0)
        assert result.lon < drone_pos.lon, "Left-half object should be west"

    def test_top_half_is_north(self, tagger, drone_pos):
        """Object on top of image → north of drone (lat increases)."""
        top_bbox = BoundingBox(x_min=0.4, y_min=0.1, x_max=0.6, y_max=0.3)
        result = tagger.pixel_to_gps(top_bbox, drone_pos, heading_deg=0.0)
        assert result.lat > drone_pos.lat, "Top-half object should be north"

    def test_bottom_half_is_south(self, tagger, drone_pos):
        """Object on bottom → south of drone (lat decreases)."""
        bot_bbox = BoundingBox(x_min=0.4, y_min=0.7, x_max=0.6, y_max=0.9)
        result = tagger.pixel_to_gps(bot_bbox, drone_pos, heading_deg=0.0)
        assert result.lat < drone_pos.lat, "Bottom-half object should be south"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Altitude-proportional offset
# ──────────────────────────────────────────────────────────────────────────────

class TestAltitudeScaling:
    def test_higher_altitude_larger_offset(self, tagger):
        """Doubling altitude should roughly double the ground offset."""
        lat, lon = 28.0, 77.0
        right_bbox = BoundingBox(x_min=0.7, y_min=0.4, x_max=0.9, y_max=0.6)
        low  = GeoPoint(lat=lat, lon=lon, alt=50.0)
        high = GeoPoint(lat=lat, lon=lon, alt=100.0)

        res_low  = tagger.pixel_to_gps(right_bbox, low)
        res_high = tagger.pixel_to_gps(right_bbox, high)

        delta_low  = abs(res_low.lon  - lon)
        delta_high = abs(res_high.lon - lon)

        # High altitude should produce ~2× the offset
        assert delta_high > delta_low * 1.5

    def test_zero_altitude_fallback(self, tagger, center_bbox):
        """Alt=0 should trigger fallback → drone position returned."""
        pos = GeoPoint(lat=28.0, lon=77.0, alt=0.0)
        result = tagger.pixel_to_gps(center_bbox, pos)
        assert result.lat == pos.lat
        assert result.lon == pos.lon

    def test_submin_altitude_fallback(self, tagger, center_bbox):
        """Alt < 1 m should trigger fallback."""
        pos = GeoPoint(lat=28.0, lon=77.0, alt=0.5)
        result = tagger.pixel_to_gps(center_bbox, pos)
        assert result.lat == pos.lat


# ──────────────────────────────────────────────────────────────────────────────
# 4. Heading rotation
# ──────────────────────────────────────────────────────────────────────────────

class TestHeadingRotation:
    def test_heading_90_right_becomes_south(self, tagger, drone_pos):
        """
        Heading=90° (East): right side of image is now South.
        With E-facing camera, image right → increasing south = decreasing lat.
        """
        right_bbox = BoundingBox(x_min=0.7, y_min=0.4, x_max=0.9, y_max=0.6)
        result = tagger.pixel_to_gps(right_bbox, drone_pos, heading_deg=90.0)
        # With heading East, right side of frame maps to southward offset
        assert result.lat < drone_pos.lat, "Heading=90, right side should be south"

    def test_heading_180_right_becomes_west(self, tagger, drone_pos):
        """Heading=180° (South): right side → west (lon decreases)."""
        right_bbox = BoundingBox(x_min=0.7, y_min=0.4, x_max=0.9, y_max=0.6)
        result = tagger.pixel_to_gps(right_bbox, drone_pos, heading_deg=180.0)
        assert result.lon < drone_pos.lon, "Heading=180, right side should be west"

    def test_heading_360_same_as_0(self, tagger, drone_pos):
        """Heading=360 should give same result as heading=0."""
        bbox = BoundingBox(x_min=0.6, y_min=0.4, x_max=0.8, y_max=0.6)
        r0   = tagger.pixel_to_gps(bbox, drone_pos, heading_deg=0.0)
        r360 = tagger.pixel_to_gps(bbox, drone_pos, heading_deg=360.0)
        assert abs(r0.lat - r360.lat) < 1e-8
        assert abs(r0.lon - r360.lon) < 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# 5. Low confidence fallback
# ──────────────────────────────────────────────────────────────────────────────

class TestConfidenceFallback:
    def test_low_confidence_returns_drone_pos(self, tagger, center_bbox):
        pos = GeoPoint(lat=28.0, lon=77.0, alt=100.0)
        result = tagger.pixel_to_gps(center_bbox, pos, confidence=0.05)
        assert result.lat == pos.lat

    def test_high_confidence_geo_tagged(self, tagger):
        pos = GeoPoint(lat=28.0, lon=77.0, alt=100.0)
        right_bbox = BoundingBox(x_min=0.7, y_min=0.4, x_max=0.9, y_max=0.6)
        result = tagger.pixel_to_gps(right_bbox, pos, confidence=0.9)
        assert result.lon != pos.lon, "High-conf right bbox should be east of drone"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Batch tagging via tag_detections
# ──────────────────────────────────────────────────────────────────────────────

def _make_detection(cx: float, cy: float) -> DetectionEvent:
    half = 0.05
    return DetectionEvent(
        source=DetectionSource.RGB,
        detection_class=DetectionClass.PERSON,
        confidence=0.85,
        bbox=BoundingBox(
            x_min=cx - half,
            y_min=cy - half,
            x_max=cx + half,
            y_max=cy + half,
        ),
        geo_position=None,
    )


class TestBatchTagging:
    def test_tag_detections_mutates_in_place(self, tagger, drone_pos):
        dets = [_make_detection(0.5, 0.5), _make_detection(0.8, 0.5)]
        result = tagger.tag_detections(dets, drone_pos, heading_deg=0.0)
        assert result is dets              # same list returned
        assert all(d.geo_position is not None for d in dets)

    def test_tag_detections_empty_list(self, tagger, drone_pos):
        result = tagger.tag_detections([], drone_pos)
        assert result == []

    def test_tag_detections_no_drone_pos(self, tagger):
        dets = [_make_detection(0.5, 0.5)]
        result = tagger.tag_detections(dets, None)  # type: ignore[arg-type]
        # Without drone position, geo_position unchanged (None)
        assert result[0].geo_position is None

    def test_different_positions_for_different_bboxes(self, tagger, drone_pos):
        det_left  = _make_detection(0.1, 0.5)
        det_right = _make_detection(0.9, 0.5)
        tagger.tag_detections([det_left, det_right], drone_pos)
        assert det_left.geo_position.lon < det_right.geo_position.lon


# ──────────────────────────────────────────────────────────────────────────────
# 7. Haversine distance helper
# ──────────────────────────────────────────────────────────────────────────────

class TestHaversineDistance:
    def test_same_point_is_zero(self):
        p = GeoPoint(lat=28.0, lon=77.0, alt=0.0)
        assert GeoTagger.haversine_distance(p, p) < 1e-3

    def test_one_degree_lat(self):
        p1 = GeoPoint(lat=28.0, lon=77.0, alt=0.0)
        p2 = GeoPoint(lat=29.0, lon=77.0, alt=0.0)
        dist = GeoTagger.haversine_distance(p1, p2)
        assert 110_000 < dist < 112_000, f"1° lat should be ≈111 km, got {dist:.0f} m"

    def test_known_distance(self):
        # Delhi ↔ Jaipur straight-line (great-circle) ≈ 235 km
        delhi  = GeoPoint(lat=28.6139, lon=77.2090, alt=0.0)
        jaipur = GeoPoint(lat=26.9124, lon=75.7873, alt=0.0)
        dist   = GeoTagger.haversine_distance(delhi, jaipur)
        assert 220_000 < dist < 260_000, f"Delhi↔Jaipur should be ~235 km, got {dist/1000:.1f} km"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Statistics
# ──────────────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_increment_on_tag(self, tagger, drone_pos, center_bbox):
        assert tagger.stats["tagged"] == 0
        tagger.pixel_to_gps(center_bbox, drone_pos, confidence=0.9)
        assert tagger.stats["tagged"] == 1

    def test_fallback_counted(self, tagger, center_bbox):
        pos = GeoPoint(lat=28.0, lon=77.0, alt=0.0)   # alt=0 triggers fallback
        assert tagger.stats["fallbacks"] == 0
        tagger.pixel_to_gps(center_bbox, pos)
        assert tagger.stats["fallbacks"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 9. Known-value accuracy test
# ──────────────────────────────────────────────────────────────────────────────

class TestKnownValues:
    def test_edge_pixel_ground_offset_matches_formula(self):
        """
        HFOV=90° → tan(45°)=1 → at altitude H, a pixel at cx_norm=0.75
        is  (0.75-0.5)*2*H*1 = 0.5*H metres east of the drone.
        At H=100 m that is 50 m ≈ 50/111111 degrees lon (equator).
        """
        tagger = GeoTagger(hfov_deg=90.0, vfov_deg=90.0)
        pos    = GeoPoint(lat=0.0, lon=0.0, alt=100.0)
        # bbox centre at cx_norm = 0.75
        bbox = BoundingBox(x_min=0.70, y_min=0.45, x_max=0.80, y_max=0.55)
        result = tagger.pixel_to_gps(bbox, pos, heading_deg=0.0)

        # Expected east offset = (0.75-0.5) * 2 * 100 * tan(45°) = 50 m
        expected_delta_lon = 50.0 / _METRES_PER_DEG_LAT
        actual_delta_lon   = result.lon - pos.lon
        assert abs(actual_delta_lon - expected_delta_lon) < 0.001 * expected_delta_lon, (
            f"cx=0.75 lon offset {actual_delta_lon:.7f}° expected ≈{expected_delta_lon:.7f}°"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 10. YOLODetector integration — geo-tagging wired in
# ──────────────────────────────────────────────────────────────────────────────

class TestYOLODetectorIntegration:
    """Verify that YOLODetector passes heading through to GeoTagger."""

    def test_mock_detect_geo_tags_with_altitude(self):
        """Mock detections at non-zero altitude should NOT all equal drone pos."""
        import random
        random.seed(42)

        from src.vision.yolo_detector import YOLODetector
        detector = YOLODetector(confidence_threshold=0.0)

        drone = GeoPoint(lat=28.6, lon=77.2, alt=80.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        non_zero_count = 0
        for _ in range(50):
            dets = detector.detect(frame, drone, heading_deg=0.0)
            for d in dets:
                if d.geo_position:
                    dist = GeoTagger.haversine_distance(
                        GeoPoint(lat=drone.lat, lon=drone.lon, alt=0.0),
                        d.geo_position,
                    )
                    # Mock detection bbox centre is random → should be offset
                    if dist > 0.1:
                        non_zero_count += 1

        assert non_zero_count > 0, "At least some detections should be offset from drone GPS"

    def test_detector_stats_include_geo_tagger(self):
        from src.vision.yolo_detector import YOLODetector
        detector = YOLODetector()
        stats = detector.stats
        assert "geo_tagger" in stats
        assert "tagged" in stats["geo_tagger"]
