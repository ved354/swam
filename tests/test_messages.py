"""
VayuSwarm — Message Protocol Tests

Tests all Pydantic message schemas for:
  - Serialization/deserialization round-trips
  - Field validation
  - FusedEvent.to_llm_text() formatting
"""

import json
import pytest
from proto.messages import (
    BaseMessage,
    BoundingBox,
    DetectionClass,
    DetectionEvent,
    DetectionSource,
    DroneReport,
    DroneState,
    DroneTelemetry,
    FusedEvent,
    GeoPoint,
    GeoZone,
    GroundCommand,
    CommandType,
    CommandPriority,
    LLMDecision,
    MissionDefinition,
    SafetyVeto,
    SafetyVetoReason,
    SwarmMessage,
    ThermalDetection,
    ThreatLevel,
    BehaviorType,
    UniformType,
)


class TestGeoPoint:
    def test_valid_point(self):
        p = GeoPoint(lat=17.385, lon=78.487, alt=50.0)
        assert p.lat == 17.385
        assert p.lon == 78.487
        assert p.alt == 50.0

    def test_boundaries(self):
        p = GeoPoint(lat=90, lon=180)
        assert p.lat == 90
        p = GeoPoint(lat=-90, lon=-180)
        assert p.lat == -90


class TestDroneTelemetry:
    def test_create_telemetry(self):
        t = DroneTelemetry(
            source_id="drone_01",
            drone_id="drone_01",
            position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            heading=90.0,
            speed=10.0,
            battery_pct=85.0,
            state=DroneState.PATROL,
        )
        assert t.drone_id == "drone_01"
        assert t.battery_pct == 85.0
        assert t.state == DroneState.PATROL
        assert t.msg_id  # auto-generated

    def test_serialization_round_trip(self):
        t = DroneTelemetry(
            source_id="drone_01",
            drone_id="drone_01",
            position=GeoPoint(lat=17.385, lon=78.487),
            heading=0,
            speed=0,
            battery_pct=100,
        )
        json_str = t.model_dump_json()
        t2 = DroneTelemetry.model_validate_json(json_str)
        assert t2.drone_id == t.drone_id
        assert t2.position.lat == t.position.lat


class TestDetectionEvent:
    def test_create_detection(self):
        d = DetectionEvent(
            source=DetectionSource.RGB,
            detection_class=DetectionClass.PERSON,
            confidence=0.87,
            bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8),
        )
        assert d.detection_class == DetectionClass.PERSON
        assert d.confidence == 0.87
        assert d.bbox.x_min == 0.1


class TestFusedEvent:
    def test_create_fused(self):
        f = FusedEvent(
            detection_class=DetectionClass.PERSON,
            class_confidence=0.87,
            armed=True,
            armed_confidence=0.73,
            weapon_class=DetectionClass.WEAPON_RIFLE,
            uniform=UniformType.MILITARY,
            uniform_confidence=0.61,
            behavior=BehaviorType.EVASIVE_MOVEMENT,
            threat_level=ThreatLevel.HIGH,
            in_ngz=True,
            sources=[DetectionSource.RGB, DetectionSource.THERMAL],
        )
        assert f.armed is True
        assert f.threat_level == ThreatLevel.HIGH

    def test_to_llm_text(self):
        f = FusedEvent(
            detection_class=DetectionClass.PERSON,
            class_confidence=0.87,
            armed=True,
            armed_confidence=0.73,
            weapon_class=DetectionClass.WEAPON_RIFLE,
            behavior=BehaviorType.EVASIVE_MOVEMENT,
            threat_level=ThreatLevel.HIGH,
            in_ngz=True,
            sources=[DetectionSource.RGB, DetectionSource.THERMAL],
        )
        text = f.to_llm_text()
        assert "person detected" in text
        assert "87%" in text
        assert "ARMED" in text
        assert "evasive_movement" in text
        assert "NO-GO ZONE" in text

    def test_serialization_round_trip(self):
        f = FusedEvent(
            detection_class=DetectionClass.VEHICLE,
            class_confidence=0.65,
            sources=[DetectionSource.THERMAL],
        )
        json_str = f.model_dump_json()
        f2 = FusedEvent.model_validate_json(json_str)
        assert f2.detection_class == DetectionClass.VEHICLE


class TestGroundCommand:
    def test_create_command(self):
        cmd = GroundCommand(
            source_id="ground",
            target_drone_id="drone_01",
            command_type=CommandType.GOTO_WAYPOINT,
            priority=CommandPriority.HIGH,
            waypoint=GeoPoint(lat=17.39, lon=78.49, alt=60),
            message="Investigate target area",
        )
        assert cmd.command_type == CommandType.GOTO_WAYPOINT
        assert cmd.priority == CommandPriority.HIGH


class TestMission:
    def test_create_mission(self):
        m = MissionDefinition(
            name="Test Mission",
            drone_ids=["drone_01", "drone_02"],
            roe={"engagement_allowed": False},
        )
        assert m.name == "Test Mission"
        assert len(m.drone_ids) == 2
        assert m.active is True


class TestSafetyVeto:
    def test_create_veto(self):
        v = SafetyVeto(
            drone_id="drone_01",
            reason=SafetyVetoReason.NGZ_VIOLATION,
            original_action="INVESTIGATE",
            override_action="HOLD",
            severity=ThreatLevel.HIGH,
            details="Target inside no-go zone",
        )
        assert v.reason == SafetyVetoReason.NGZ_VIOLATION
        assert v.override_action == "HOLD"
