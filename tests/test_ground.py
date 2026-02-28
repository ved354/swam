"""
VayuSwarm — Ground Station Tests
"""

import pytest
from proto.messages import (
    DroneReport,
    DroneState,
    DroneTelemetry,
    GeoPoint,
)
from src.ground.fleet_manager import FleetManager
from src.ground.mission_planner import MissionPlanner


class TestFleetManager:
    def test_register_drone(self):
        fleet = FleetManager()
        fleet.register_drone("drone_01")
        assert fleet.drone_count == 1

    def test_update_drone(self):
        fleet = FleetManager()
        fleet.register_drone("drone_01")

        report = DroneReport(
            source_id="drone_01",
            drone_id="drone_01",
            telemetry=DroneTelemetry(
                source_id="drone_01",
                drone_id="drone_01",
                position=GeoPoint(lat=17.385, lon=78.487, alt=50),
                heading=90,
                speed=10,
                battery_pct=85,
                state=DroneState.PATROL,
            ),
        )
        fleet.update_drone(report)

        status = fleet.get_drone_status("drone_01")
        assert status is not None
        assert status.online
        assert status.state == DroneState.PATROL
        assert status.battery == 85

    def test_low_battery_warning(self):
        fleet = FleetManager()
        report = DroneReport(
            source_id="drone_01",
            drone_id="drone_01",
            telemetry=DroneTelemetry(
                source_id="drone_01",
                drone_id="drone_01",
                position=GeoPoint(lat=17.385, lon=78.487),
                heading=0,
                speed=0,
                battery_pct=15,
            ),
        )
        fleet.update_drone(report)
        status = fleet.get_drone_status("drone_01")
        assert any("LOW_BATTERY" in w for w in status.warnings)

    def test_fleet_summary(self):
        fleet = FleetManager()
        fleet.register_drone("drone_01")
        fleet.register_drone("drone_02")
        summary = fleet.get_fleet_summary()
        assert len(summary) == 2


class TestMissionPlanner:
    def test_create_mission(self):
        planner = MissionPlanner()
        mission = planner.create_mission(
            name="Test Mission",
            drone_ids=["drone_01"],
            objectives=[{"description": "Test objective"}],
        )
        assert mission.name == "Test Mission"
        assert planner.active_mission is not None

    def test_default_mission(self):
        planner = MissionPlanner()
        mission = planner.create_default_mission(["drone_01", "drone_02"])
        assert mission is not None
        assert len(mission.drone_ids) == 2
        assert len(mission.patrol_zones) > 0
        assert len(mission.no_go_zones) > 0

    def test_complete_objective(self):
        planner = MissionPlanner()
        mission = planner.create_mission(
            name="Test",
            drone_ids=["drone_01"],
            objectives=[{"description": "Objective 1"}],
        )
        obj_id = mission.objectives[0].objective_id
        assert planner.complete_objective(obj_id)
        assert mission.objectives[0].completed

    def test_mission_progress(self):
        planner = MissionPlanner()
        planner.create_mission(
            name="Test",
            drone_ids=["drone_01"],
            objectives=[
                {"description": "Task 1"},
                {"description": "Task 2"},
            ],
        )
        progress = planner.progress
        assert progress["active"]
        assert progress["objectives_total"] == 2
        assert progress["objectives_completed"] == 0
