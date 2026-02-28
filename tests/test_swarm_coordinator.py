"""
Tests for SwarmCoordinator — zone assignment, rebalancing,
target handoff, collision avoidance, and swarm consensus.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from proto.messages import (
    CommandType,
    DroneState,
    GeoPoint,
    GeoZone,
    GroundCommand,
    ThreatLevel,
)
from src.ground.swarm_coordinator import (
    CollisionAvoider,
    SwarmConsensus,
    SwarmCoordinator,
    TrackedTarget,
    ZoneDivider,
    _haversine,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _zone(
    lat_s=17.380, lat_e=17.390, lon_s=78.480, lon_e=78.495
) -> GeoZone:
    return GeoZone(
        zone_id="z1",
        name="Test Zone",
        points=[
            GeoPoint(lat=lat_s, lon=lon_s, alt=60),
            GeoPoint(lat=lat_s, lon=lon_e, alt=60),
            GeoPoint(lat=lat_e, lon=lon_e, alt=60),
            GeoPoint(lat=lat_e, lon=lon_s, alt=60),
        ],
    )


def _pos(lat: float, lon: float, alt: float = 60.0) -> GeoPoint:
    return GeoPoint(lat=lat, lon=lon, alt=alt)


def _mock_fleet(
    drone_ids: list[str],
    positions: dict[str, GeoPoint] | None = None,
    batteries: dict[str, float] | None = None,
    states: dict[str, DroneState] | None = None,
):
    """Build a minimal mock FleetManager."""
    fleet = MagicMock()
    fleet.get_online_drone_ids.return_value = drone_ids
    fleet.get_all_positions.return_value = positions or {d: _pos(17.385, 78.487) for d in drone_ids}
    fleet.get_all_reports.return_value = {}

    def _status(drone_id):
        s = MagicMock()
        s.battery = (batteries or {}).get(drone_id, 90.0)
        telem = MagicMock()
        telem.state = (states or {}).get(drone_id, DroneState.PATROL)
        s.last_telemetry = telem
        return s

    fleet.get_drone_status.side_effect = _status
    return fleet


# ─── Haversine ────────────────────────────────────────────────────────────────

def test_haversine_zero():
    p = _pos(17.385, 78.487)
    assert _haversine(p, p) == pytest.approx(0.0, abs=0.01)


def test_haversine_known():
    # ~1 degree lat ≈ 111 km
    a = _pos(0.0, 0.0)
    b = _pos(1.0, 0.0)
    assert 110_000 < _haversine(a, b) < 112_000


# ─── ZoneDivider ──────────────────────────────────────────────────────────────

def test_zone_divider_single_drone():
    sectors = ZoneDivider.divide(_zone().points, 1)
    assert len(sectors) == 1
    assert sectors[0].sector_id == "sector_00"
    assert len(sectors[0].waypoints) > 0


def test_zone_divider_three_drones():
    sectors = ZoneDivider.divide(_zone().points, 3)
    assert len(sectors) == 3
    # All centers should be distinct
    centers = [(s.center.lat, s.center.lon) for s in sectors]
    assert len(set(centers)) == 3


def test_zone_divider_lawnmower_waypoints():
    sectors = ZoneDivider.divide(_zone().points, 2)
    for s in sectors:
        assert len(s.waypoints) >= 2


# ─── CollisionAvoider ─────────────────────────────────────────────────────────

def test_no_deflection_when_safe():
    proposed = _pos(17.385, 78.480)
    others   = {"drone_02": _pos(17.390, 78.490)}   # ~1200 m away
    result   = CollisionAvoider.check_and_deflect("drone_01", proposed, others)
    assert result.lat == pytest.approx(proposed.lat, abs=1e-7)
    assert result.lon == pytest.approx(proposed.lon, abs=1e-7)


def test_deflection_when_too_close():
    proposed = _pos(17.3850, 78.4870)
    # 2 metres away — same position basically
    others   = {"drone_02": _pos(17.3850, 78.4870)}
    result   = CollisionAvoider.check_and_deflect("drone_01", proposed, others, safe_dist_m=30.0)
    dist     = _haversine(result, others["drone_02"])
    assert dist >= 30.0


def test_deflection_excludes_self():
    proposed = _pos(17.385, 78.487)
    others   = {"drone_01": _pos(17.385, 78.487)}   # same drone, same pos
    result   = CollisionAvoider.check_and_deflect("drone_01", proposed, others)
    assert result.lat == pytest.approx(proposed.lat, abs=1e-7)


# ─── SwarmConsensus ───────────────────────────────────────────────────────────

def _make_report(drone_id: str, gps: GeoPoint, threat=ThreatLevel.HIGH):
    event = MagicMock()
    event.threat_level   = threat
    event.gps_position   = gps
    report = MagicMock()
    report.drone_id      = drone_id
    report.fused_events  = [event]
    return report


def test_consensus_elects_closest_drone():
    target_pos = _pos(17.385, 78.487)
    # drone_01 is 50 m away, drone_02 is 200 m away — drone_01 should be elected
    positions = {
        "drone_01": _pos(17.3854, 78.487),   # ~50 m north
        "drone_02": _pos(17.387,  78.487),   # ~200 m north
    }
    reports = {
        "drone_01": _make_report("drone_01", target_pos),
        "drone_02": _make_report("drone_02", target_pos),
    }
    commands, targets = SwarmConsensus.resolve(reports, positions, {})
    track_cmds = [c for c in commands if c.command_type == CommandType.TRACK_TARGET]
    assert len(track_cmds) == 1
    assert track_cmds[0].target_drone_id == "drone_01"


def test_consensus_no_high_threat_no_commands():
    target_pos = _pos(17.385, 78.487)
    positions  = {"drone_01": _pos(17.385, 78.487)}
    reports    = {"drone_01": _make_report("drone_01", target_pos, threat=ThreatLevel.LOW)}
    commands, targets = SwarmConsensus.resolve(reports, positions, {})
    assert commands == []


def test_consensus_registers_target():
    target_pos = _pos(17.385, 78.487)
    positions  = {"drone_01": _pos(17.385, 78.487)}
    reports    = {"drone_01": _make_report("drone_01", target_pos)}
    _, targets = SwarmConsensus.resolve(reports, positions, {})
    assert len(targets) == 1


# ─── SwarmCoordinator — zone initialization ───────────────────────────────────

def test_coordinator_initializes_zones():
    coord      = SwarmCoordinator()
    drone_ids  = ["d1", "d2", "d3"]
    coord.initialize_zones(_zone(), drone_ids)
    assignment = coord.get_sector_assignment()
    assert set(assignment.keys()) == set(drone_ids)


def test_coordinator_sector_count_matches_drones():
    coord = SwarmCoordinator()
    coord.initialize_zones(_zone(), ["d1", "d2", "d3", "d4"])
    assert len(coord.get_sector_assignment()) == 4


# ─── SwarmCoordinator — rebalancing ───────────────────────────────────────────

def test_rebalance_reassigns_orphaned_sector():
    coord     = SwarmCoordinator()
    all_ids   = ["d1", "d2", "d3"]
    coord.initialize_zones(_zone(), all_ids)
    coord._last_rebalance = 0  # force rebalance

    # Simulate d1 going offline — only d2, d3 online
    fleet = _mock_fleet(["d2", "d3"])
    fleet.get_all_reports.return_value = {}

    cmds = coord.tick(fleet)
    assignment = coord.get_sector_assignment()
    # d1's sector should now be assigned to d2 or d3
    assert "d1" not in assignment.values()


# ─── SwarmCoordinator — target handoff ────────────────────────────────────────

def test_handoff_triggered_on_low_battery():
    coord    = SwarmCoordinator()
    drone_ids = ["d1", "d2"]
    coord.initialize_zones(_zone(), drone_ids)

    # d1 is tracking at 20% battery, d2 is healthy at 90%
    target = TrackedTarget(
        target_id="tgt_001",
        position=_pos(17.385, 78.487),
        threat=ThreatLevel.HIGH,
        tracking_drone="d1",
    )
    coord._active_targets["tgt_001"] = target

    positions = {"d1": _pos(17.385, 78.487), "d2": _pos(17.386, 78.487)}
    fleet = _mock_fleet(
        drone_ids,
        positions=positions,
        batteries={"d1": 20.0, "d2": 90.0},
        states={"d1": DroneState.TRACK, "d2": DroneState.PATROL},
    )

    # Mock d2 report to show it's not tracking
    d2_report = MagicMock()
    d2_report.fused_events = []
    d2_telem = MagicMock()
    d2_telem.state = DroneState.PATROL
    d2_report.telemetry = d2_telem
    fleet.get_all_reports.return_value = {"d1": MagicMock(fused_events=[]), "d2": d2_report}

    cmds = coord._process_handoffs(fleet, positions, fleet.get_all_reports())
    track_cmds = [c for c in cmds if c.command_type == CommandType.TRACK_TARGET]
    rtl_cmds   = [c for c in cmds if c.command_type == CommandType.RETURN_TO_LAUNCH]

    assert len(track_cmds) == 1
    assert track_cmds[0].target_drone_id == "d2"
    assert len(rtl_cmds) == 1
    assert rtl_cmds[0].target_drone_id == "d1"


def test_no_handoff_when_battery_healthy():
    coord     = SwarmCoordinator()
    drone_ids = ["d1", "d2"]
    coord.initialize_zones(_zone(), drone_ids)

    target = TrackedTarget(
        target_id="tgt_002",
        position=_pos(17.385, 78.487),
        threat=ThreatLevel.HIGH,
        tracking_drone="d1",
    )
    coord._active_targets["tgt_002"] = target

    positions = {"d1": _pos(17.385, 78.487), "d2": _pos(17.386, 78.487)}
    fleet = _mock_fleet(drone_ids, positions=positions, batteries={"d1": 80.0, "d2": 90.0})
    fleet.get_all_reports.return_value = {}

    cmds = coord._process_handoffs(fleet, positions, {})
    assert cmds == []


# ─── SwarmCoordinator — collision avoidance in tick ───────────────────────────

def test_tick_returns_list():
    coord = SwarmCoordinator()
    coord.initialize_zones(_zone(), ["d1"])
    fleet = _mock_fleet(["d1"])
    result = coord.tick(fleet)
    assert isinstance(result, list)


def test_active_targets_empty_initially():
    coord = SwarmCoordinator()
    assert coord.get_active_targets() == []
