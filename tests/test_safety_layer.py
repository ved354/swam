"""
VayuSwarm — Safety Layer Tests

Tests the two-tier safety system:
  - Battery critical → force RTL
  - No-go zone enforcement
  - Altitude limits
  - Geofence breach
  - Collision avoidance
  - ROE compliance
  - Anomaly detection
"""

import pytest
from proto.messages import (
    DroneState,
    GeoPoint,
    GeoZone,
    LLMDecision,
    SafetyVetoReason,
    ThreatLevel,
)
from src.drone.safety_layer import SafetyLayer


@pytest.fixture
def safety():
    """Create a safety layer instance with test config."""
    home = GeoPoint(lat=17.385, lon=78.487, alt=0)
    ngz = GeoZone(
        zone_id="ngz_test",
        name="Test No-Go Zone",
        points=[
            GeoPoint(lat=17.390, lon=78.490),
            GeoPoint(lat=17.390, lon=78.492),
            GeoPoint(lat=17.388, lon=78.492),
            GeoPoint(lat=17.388, lon=78.490),
        ],
        is_no_go=True,
    )
    return SafetyLayer(
        drone_id="drone_test",
        home_position=home,
        max_altitude_m=120.0,
        min_altitude_m=5.0,
        battery_critical_pct=15.0,
        collision_radius_m=10.0,
        geofence_radius_m=5000.0,
        no_go_zones=[ngz],
    )


class TestBatterySafety:
    def test_critical_battery_forces_rtl(self, safety):
        decision = LLMDecision(source="local_llm", action="PATROL", confidence=0.8)
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=10.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.BATTERY_CRITICAL
        assert result.action == "RTL"

    def test_normal_battery_passes(self, safety):
        decision = LLMDecision(source="local_llm", action="PATROL", confidence=0.8)
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is None
        assert result.action == "PATROL"

    def test_rtl_with_critical_battery_passes(self, safety):
        decision = LLMDecision(source="local_llm", action="RTL", confidence=0.9)
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=10.0,
        )
        assert veto is None  # Already doing RTL


class TestNoGoZone:
    def test_waypoint_in_ngz_blocked(self, safety):
        decision = LLMDecision(
            source="local_llm",
            action="INVESTIGATE",
            confidence=0.8,
            suggested_waypoint=GeoPoint(lat=17.389, lon=78.491, alt=50),
        )
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.NGZ_VIOLATION

    def test_waypoint_outside_ngz_passes(self, safety):
        decision = LLMDecision(
            source="local_llm",
            action="GOTO",
            confidence=0.8,
            suggested_waypoint=GeoPoint(lat=17.386, lon=78.488, alt=50),
        )
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is None


class TestAltitude:
    def test_altitude_too_high(self, safety):
        decision = LLMDecision(
            source="local_llm",
            action="GOTO",
            confidence=0.8,
            suggested_waypoint=GeoPoint(lat=17.385, lon=78.487, alt=200),
        )
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.ALTITUDE_LIMIT

    def test_altitude_too_low(self, safety):
        decision = LLMDecision(
            source="local_llm",
            action="GOTO",
            confidence=0.8,
            suggested_waypoint=GeoPoint(lat=17.385, lon=78.487, alt=2),
        )
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.ALTITUDE_LIMIT


class TestROECompliance:
    def test_engage_action_blocked(self, safety):
        decision = LLMDecision(source="local_llm", action="ENGAGE", confidence=0.9)
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.TRACK,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.ROE_VIOLATION

    def test_fire_action_blocked(self, safety):
        decision = LLMDecision(source="local_llm", action="FIRE", confidence=0.9)
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.TRACK,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.ROE_VIOLATION


class TestCollision:
    def test_collision_risk_blocked(self, safety):
        decision = LLMDecision(
            source="local_llm",
            action="GOTO",
            confidence=0.8,
            suggested_waypoint=GeoPoint(lat=17.385, lon=78.487, alt=50),
        )
        peer_positions = {
            "drone_02": GeoPoint(lat=17.385, lon=78.487, alt=50),
        }
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.384, lon=78.486, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
            peer_positions=peer_positions,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.COLLISION_RISK


class TestGeofence:
    def test_geofence_breach(self, safety):
        decision = LLMDecision(
            source="local_llm",
            action="GOTO",
            confidence=0.8,
            suggested_waypoint=GeoPoint(lat=18.0, lon=79.0, alt=50),  # Far away
        )
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason == SafetyVetoReason.GEOFENCE_BREACH
