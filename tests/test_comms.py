"""
VayuSwarm — Communication Layer Tests
"""

import pytest
from proto.messages import (
    DroneTelemetry,
    DroneState,
    GeoPoint,
    GroundCommand,
    CommandType,
    CommandPriority,
)
from src.comms.serializer import MessageSerializer, build_message_registry


class TestSerializer:
    def test_serialize_deserialize(self):
        registry = build_message_registry()
        msg = DroneTelemetry(
            source_id="drone_01",
            drone_id="drone_01",
            position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            heading=90.0,
            speed=10.0,
            battery_pct=85.0,
        )

        frames = MessageSerializer.serialize(msg, "drone/drone_01/report")
        assert len(frames) == 3
        assert frames[0] == b"drone/drone_01/report"
        assert frames[1] == b"DroneTelemetry"

        topic, deserialized = MessageSerializer.deserialize(frames, registry)
        assert topic == "drone/drone_01/report"
        assert isinstance(deserialized, DroneTelemetry)
        assert deserialized.drone_id == "drone_01"

    def test_serialize_simple(self):
        registry = build_message_registry()
        cmd = GroundCommand(
            source_id="ground",
            target_drone_id="drone_01",
            command_type=CommandType.GOTO_WAYPOINT,
            priority=CommandPriority.HIGH,
            waypoint=GeoPoint(lat=17.39, lon=78.49, alt=60),
        )

        data = MessageSerializer.serialize_simple(cmd)
        assert isinstance(data, bytes)

        deserialized = MessageSerializer.deserialize_simple(data, registry)
        assert isinstance(deserialized, GroundCommand)
        assert deserialized.target_drone_id == "drone_01"

    def test_unknown_type_raises(self):
        registry = build_message_registry()
        frames = [b"topic", b"UnknownType", b"{}"]
        with pytest.raises(ValueError, match="Unknown message type"):
            MessageSerializer.deserialize(frames, registry)


class TestMessageRegistry:
    def test_all_types_registered(self):
        registry = build_message_registry()
        expected = [
            "DroneTelemetry", "DetectionEvent", "ThermalDetection",
            "FusedEvent", "DroneReport", "GroundCommand",
            "SwarmMessage", "SafetyVeto", "MissionDefinition", "LLMDecision",
        ]
        for name in expected:
            assert name in registry, f"Missing: {name}"
