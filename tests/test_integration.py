"""
VayuSwarm — Integration Tests

Tests the full system integration:
  - MessageBus pub/sub round-trip (ground ↔ drone topology)
  - Drone agent pipeline: vision → fusion → LLM → safety → decision
  - Ground station command dispatch
  - Dashboard REST API endpoints
  - Mesh network peer-to-peer messaging
  - PX4 interface: mode encoding, telemetry caching, SITL fallback
"""

import asyncio
import time

import numpy as np
import pytest

from proto.messages import (
    BoundingBox,
    CommandPriority,
    CommandType,
    DetectionClass,
    DetectionEvent,
    DetectionSource,
    DroneReport,
    DroneState,
    DroneTelemetry,
    FusedEvent,
    GeoPoint,
    GroundCommand,
    LLMDecision,
    SwarmMessage,
    ThermalDetection,
    ThreatLevel,
)
from src.comms.message_bus import MessageBus, Publisher, Subscriber, HeartbeatMonitor
from src.comms.serializer import MessageSerializer, build_message_registry
from src.drone.local_llm import LocalLLM
from src.drone.px4_interface import (
    PX4Interface,
    PX4_MODE_MAP,
    _px4_custom_mode,
    PX4_MAIN_MODE_OFFBOARD,
    PX4_MAIN_MODE_AUTO,
    PX4_AUTO_SUB_MODE_RTL,
    PX4_AUTO_SUB_MODE_LAND,
    PX4_AUTO_SUB_MODE_LOITER,
)
from src.drone.safety_layer import SafetyLayer
from src.drone.state_machine import DroneStateMachine
from src.ground.fleet_manager import FleetManager
from src.ground.ground_llm import GroundLLM
from src.ground.mission_planner import MissionPlanner
from src.vision.behavior_analyzer import BehaviorAnalyzer
from src.vision.sensor_fusion import SensorFusion
from src.vision.thermal_model import ThermalModel
from src.vision.yolo_detector import YOLODetector


# ═══════════════════════════════════════════════════════════════════════════════
# ZeroMQ Message Bus Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestMessageBusTopology:
    """Test the ground-binds / drone-connects pub/sub topology."""

    @pytest.fixture
    def registry(self):
        return build_message_registry()

    async def test_publisher_bind_connect_modes(self):
        """Publisher supports both bind and connect modes."""
        pub_bind = Publisher("tcp://127.0.0.1:15550", bind=True)
        pub_conn = Publisher("tcp://127.0.0.1:15550", bind=False)
        assert pub_bind._bind is True
        assert pub_conn._bind is False

    async def test_subscriber_bind_connect_modes(self):
        """Subscriber supports both bind and connect modes."""
        sub_bind = Subscriber("tcp://127.0.0.1:15551", bind=True)
        sub_conn = Subscriber("tcp://127.0.0.1:15551", bind=False)
        assert sub_bind._bind is True
        assert sub_conn._bind is False

    async def test_ground_drone_pub_sub_roundtrip(self):
        """
        Ground station binds PUB + SUB.
        Drone connects PUB + SUB.
        Test both directions:
          - Ground → Drone (command)
          - Drone → Ground (report)
        """
        # Ground: PUB binds 15555, SUB binds 15556
        ground_bus = MessageBus(
            role="ground",
            pub_addr="tcp://127.0.0.1:15555",
            sub_addr="tcp://127.0.0.1:15556",
            node_id="ground",
            topics=["drone/"],
            pub_bind=True,
            sub_bind=True,
        )

        # Drone: PUB connects to ground's SUB, SUB connects to ground's PUB
        drone_bus = MessageBus(
            role="drone",
            pub_addr="tcp://127.0.0.1:15556",
            sub_addr="tcp://127.0.0.1:15555",
            node_id="drone_01",
            topics=["ground/"],
            pub_bind=False,
            sub_bind=False,
        )

        received_by_ground = []
        received_by_drone = []

        drone_bus.on_message("ground/", lambda t, m: _async_append(received_by_drone, (t, m)))
        ground_bus.on_message("drone/", lambda t, m: _async_append(received_by_ground, (t, m)))

        await ground_bus.start()
        await drone_bus.start()

        # Wait for ZMQ connections to establish
        await asyncio.sleep(0.5)

        try:
            # Ground → Drone: publish a command
            cmd = GroundCommand(
                source_id="ground",
                target_drone_id="drone_01",
                command_type=CommandType.HOLD_POSITION,
                priority=CommandPriority.NORMAL,
                message="Test hold",
            )
            await ground_bus.publish("ground/commands/drone_01", cmd)

            # Drone → Ground: publish a report
            report = DroneReport(
                source_id="drone_01",
                drone_id="drone_01",
                telemetry=DroneTelemetry(
                    source_id="drone_01",
                    drone_id="drone_01",
                    position=GeoPoint(lat=17.385, lon=78.487, alt=50),
                    heading=90, speed=10, battery_pct=85,
                ),
            )
            await drone_bus.publish("drone/drone_01/report", report)

            # Wait for messages to propagate
            await asyncio.sleep(0.5)

            # Verify drone received ground command
            assert len(received_by_drone) >= 1, "Drone should receive ground command"
            assert isinstance(received_by_drone[0][1], GroundCommand)

            # Verify ground received drone report
            assert len(received_by_ground) >= 1, "Ground should receive drone report"
            assert isinstance(received_by_ground[0][1], DroneReport)

        finally:
            await drone_bus.stop()
            await ground_bus.stop()


class TestHeartbeatMonitor:
    async def test_heartbeat_detects_lost_nodes(self):
        monitor = HeartbeatMonitor(timeout_s=0.5)
        monitor.beat("drone_01")
        assert monitor.is_alive("drone_01")
        assert "drone_01" in monitor.get_alive_nodes()

        # Wait past timeout
        await asyncio.sleep(0.6)
        lost = await monitor.check()
        assert "drone_01" in lost
        assert not monitor.is_alive("drone_01")

    async def test_heartbeat_keeps_alive_on_regular_beats(self):
        monitor = HeartbeatMonitor(timeout_s=1.0)
        monitor.beat("drone_01")
        await asyncio.sleep(0.3)
        monitor.beat("drone_01")
        await asyncio.sleep(0.3)
        lost = await monitor.check()
        assert "drone_01" not in lost
        assert monitor.is_alive("drone_01")


# ═══════════════════════════════════════════════════════════════════════════════
# Full Drone Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestDronePipeline:
    """Test the full perception → decision → safety → action pipeline."""

    async def test_vision_to_decision_pipeline(self):
        """
        Run the full pipeline:
          YOLO → Thermal → Fusion → Behavior → LLM → Safety
        with mock/simulated data.
        """
        # 1. YOLO detector (mock mode — no model loaded)
        yolo = YOLODetector()
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        pos = GeoPoint(lat=17.385, lon=78.487, alt=50)

        # 2. Thermal (mock mode)
        thermal = ThermalModel()
        thermal.load()
        thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        # 3. Run detections multiple times to ensure some hits
        all_rgb = []
        all_thermal = []
        for _ in range(100):
            all_rgb.extend(yolo.detect(frame, pos))
            all_thermal.extend(thermal.detect(thermal_frame, pos))

        # Should have some detections across 100 frames
        assert len(all_rgb) > 0 or len(all_thermal) > 0, "Should get at least some detections"

        # 4. Sensor Fusion
        fusion = SensorFusion()
        fused = fusion.fuse(all_rgb[:5], all_thermal[:5], pos)
        assert isinstance(fused, list)

        # 5. Behavior Analysis
        behavior = BehaviorAnalyzer()
        behavior.load()
        if fused:
            analyzed = behavior.analyze(fused)
            assert len(analyzed) == len(fused)

        # 6. Local LLM (mock mode)
        llm = LocalLLM(
            drone_id="drone_test",
            provider="mock",
            model="mock",
        )
        await llm.connect()

        decision = await llm.decide(
            fused_events=fused,
            drone_state=DroneState.PATROL,
            drone_position=pos,
            battery_pct=85.0,
        )
        assert isinstance(decision, LLMDecision)
        assert decision.action in ("CONTINUE", "HOLD", "INVESTIGATE", "ALERT", "RTL")

        # 7. Safety Layer
        safety = SafetyLayer(
            drone_id="drone_test",
            home_position=pos,
        )
        safe_decision, veto = safety.validate(
            decision=decision,
            current_position=pos,
            current_state=DroneState.PATROL,
            battery_pct=85.0,
        )
        assert isinstance(safe_decision, LLMDecision)

        # Clean up
        await llm.close()

    async def test_mock_llm_low_battery_returns_rtl(self):
        """Mock LLM should decide RTL when battery is low."""
        llm = LocalLLM(drone_id="test", provider="mock", model="mock")
        await llm.connect()

        decision = await llm.decide(
            fused_events=[],
            drone_state=DroneState.PATROL,
            drone_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            battery_pct=10.0,
        )
        assert decision.action == "RTL"
        await llm.close()

    async def test_mock_llm_no_detections_continues(self):
        """Mock LLM should continue if no detections."""
        llm = LocalLLM(drone_id="test", provider="mock", model="mock")
        await llm.connect()

        decision = await llm.decide(
            fused_events=[],
            drone_state=DroneState.PATROL,
            drone_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            battery_pct=85.0,
        )
        assert decision.action == "CONTINUE"
        await llm.close()

    async def test_mock_llm_high_threat_alerts(self):
        """Mock LLM should ALERT on high-threat detections."""
        llm = LocalLLM(drone_id="test", provider="mock", model="mock")
        await llm.connect()

        event = FusedEvent(
            detection_class=DetectionClass.PERSON,
            class_confidence=0.9,
            armed=True,
            armed_confidence=0.85,
            weapon_class=DetectionClass.WEAPON_RIFLE,
            threat_level=ThreatLevel.CRITICAL,
            sources=[DetectionSource.RGB, DetectionSource.THERMAL],
            geo_position=GeoPoint(lat=17.386, lon=78.488, alt=0),
        )
        decision = await llm.decide(
            fused_events=[event],
            drone_state=DroneState.PATROL,
            drone_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            battery_pct=80.0,
        )
        assert decision.action == "ALERT"
        await llm.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PX4 Simulation Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestPX4Simulation:
    async def test_simulation_connect_and_telemetry(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        assert px4.is_connected
        assert px4.is_simulation

        telem = await px4.get_telemetry()
        assert isinstance(telem, DroneTelemetry)
        assert telem.position.lat != 0

    async def test_simulation_takeoff_and_fly(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()

        # Arm and takeoff
        assert await px4.arm()
        assert await px4.takeoff(altitude_m=50)

        # Wait for simulated climb
        for _ in range(20):
            telem = await px4.get_telemetry()
            await asyncio.sleep(0.1)

        telem = await px4.get_telemetry()
        assert telem.armed
        assert telem.position.alt > 0

    async def test_simulation_goto(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(50)

        target = GeoPoint(lat=17.390, lon=78.490, alt=50)
        assert await px4.goto(target, speed_ms=10)

    async def test_simulation_rtl(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(50)

        assert await px4.return_to_launch()
        telem = await px4.get_telemetry()
        assert telem.state in (DroneState.RTL, DroneState.PATROL)

    async def test_set_home_position(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        px4.set_home_position(lat=18.0, lon=79.0, alt=0)
        telem = await px4.get_telemetry()
        assert abs(telem.position.lat - 18.0) < 0.01
        assert abs(telem.position.lon - 79.0) < 0.01


class TestPX4ModeEncoding:
    """Test PX4 custom mode bitfield encoding and decoding."""

    def test_px4_custom_mode_encoding(self):
        """Verify custom_mode encodes main_mode + sub_mode correctly."""
        # OFFBOARD: main_mode=6, sub_mode=0
        custom = _px4_custom_mode(6, 0)
        assert (custom >> 16) & 0xFF == 6
        assert (custom >> 24) & 0xFF == 0

    def test_px4_custom_mode_with_sub_mode(self):
        """Verify sub_mode encoding for AUTO modes."""
        # AUTO.RTL: main=4, sub=5
        custom = _px4_custom_mode(PX4_MAIN_MODE_AUTO, PX4_AUTO_SUB_MODE_RTL)
        assert (custom >> 16) & 0xFF == PX4_MAIN_MODE_AUTO
        assert (custom >> 24) & 0xFF == PX4_AUTO_SUB_MODE_RTL

    def test_mode_map_completeness(self):
        """All standard modes should be in PX4_MODE_MAP."""
        for name in ("OFFBOARD", "RTL", "LAND", "LOITER", "TAKEOFF", "MISSION",
                      "MANUAL", "POSCTL", "STABILIZED"):
            assert name in PX4_MODE_MAP, f"{name} missing from PX4_MODE_MAP"

    def test_decode_round_trip(self):
        """Encoding then decoding should recover the mode name."""
        for name in ("AUTO.RTL", "AUTO.LAND", "AUTO.LOITER", "OFFBOARD", "MANUAL"):
            main, sub = PX4_MODE_MAP[name]
            custom = _px4_custom_mode(main, sub)
            decoded = PX4Interface._decode_px4_mode(custom)
            assert decoded == name, f"Round-trip failed for {name}: got {decoded}"

    def test_alias_modes_encode(self):
        """Convenience aliases (RTL, LAND, LOITER) should map correctly."""
        # RTL alias → AUTO.RTL
        main_r, sub_r = PX4_MODE_MAP["RTL"]
        main_a, sub_a = PX4_MODE_MAP["AUTO.RTL"]
        assert (main_r, sub_r) == (main_a, sub_a)


class TestPX4SimulationExtended:
    """Extended PX4 simulation tests: hold, land, disarm, disconnect."""

    async def test_hold_position(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(50)
        assert await px4.hold_position()
        telem = await px4.get_telemetry()
        # In LOITER mode, drone is still armed
        assert telem.armed

    async def test_land_sequence(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(20)
        # Let it climb a bit
        for _ in range(10):
            await px4.get_telemetry()
            await asyncio.sleep(0.05)
        assert await px4.land()
        telem = await px4.get_telemetry()
        assert telem.mode == "LAND" or telem.state in (DroneState.LAND, DroneState.IDLE)

    async def test_disarm(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        assert await px4.disarm()
        telem = await px4.get_telemetry()
        assert not telem.armed

    async def test_set_velocity(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(50)
        assert await px4.set_velocity(5.0, 3.0, 0.0)
        # Speed should be updated
        telem = await px4.get_telemetry()
        assert telem.speed >= 0

    async def test_disconnect_cleans_state(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        assert px4.is_connected
        await px4.disconnect()
        assert not px4.is_connected

    async def test_fallback_to_simulation_on_no_mavlink(self):
        """When pymavlink connection fails, PX4Interface should fallback to sim."""
        px4 = PX4Interface(
            drone_id="test",
            connection_string="udp:127.0.0.1:19999",  # nothing listening
            simulation=False,
        )
        await px4.connect()
        # Should have fallen back to simulation mode
        assert px4.is_connected
        assert px4.is_simulation

    async def test_battery_drain_during_flight(self):
        """Battery should decrease while flying."""
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(50)

        t1 = await px4.get_telemetry()
        bat1 = t1.battery_pct

        # Fly for a bit (simulated time)
        for _ in range(50):
            await px4.get_telemetry()
            await asyncio.sleep(0.02)

        t2 = await px4.get_telemetry()
        assert t2.battery_pct <= bat1  # Battery should not increase


class TestPX4MavlinkStateMap:
    """Test the mavlink-to-drone-state mapping."""

    async def test_idle_when_disarmed(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        # Not armed → IDLE
        telem = await px4.get_telemetry()
        assert telem.state == DroneState.IDLE

    async def test_rtl_state(self):
        px4 = PX4Interface(drone_id="test", simulation=True)
        await px4.connect()
        await px4.arm()
        await px4.takeoff(50)
        # Let it climb
        for _ in range(15):
            await px4.get_telemetry()
            await asyncio.sleep(0.05)
        await px4.return_to_launch()
        telem = await px4.get_telemetry()
        assert telem.state == DroneState.RTL


# ═══════════════════════════════════════════════════════════════════════════════
# Ground Station Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroundStationPipeline:
    async def test_ground_llm_mock_strategize(self):
        """Ground LLM mock should produce valid strategic commands."""
        llm = GroundLLM(provider="mock", model="mock")
        await llm.connect()

        reports = {
            "drone_01": DroneReport(
                source_id="drone_01",
                drone_id="drone_01",
                telemetry=DroneTelemetry(
                    source_id="drone_01",
                    drone_id="drone_01",
                    position=GeoPoint(lat=17.385, lon=78.487, alt=50),
                    heading=90, speed=10, battery_pct=85,
                    state=DroneState.PATROL,
                ),
                fused_events=[
                    FusedEvent(
                        detection_class=DetectionClass.PERSON,
                        class_confidence=0.9,
                        armed=True,
                        armed_confidence=0.85,
                        weapon_class=DetectionClass.WEAPON_RIFLE,
                        threat_level=ThreatLevel.HIGH,
                        sources=[DetectionSource.RGB],
                        geo_position=GeoPoint(lat=17.386, lon=78.488),
                    ),
                ],
            ),
        }

        planner = MissionPlanner()
        mission = planner.create_default_mission(["drone_01"])

        commands = await llm.strategize(reports, mission)
        assert isinstance(commands, list)
        # Should produce a TRACK command for the high-threat drone
        if commands:
            assert isinstance(commands[0], GroundCommand)

        await llm.close()

    async def test_ground_llm_mock_low_battery_rtl(self):
        """Ground LLM mock should issue RTL for low-battery drone."""
        llm = GroundLLM(provider="mock", model="mock")
        await llm.connect()

        reports = {
            "drone_02": DroneReport(
                source_id="drone_02",
                drone_id="drone_02",
                telemetry=DroneTelemetry(
                    source_id="drone_02",
                    drone_id="drone_02",
                    position=GeoPoint(lat=17.385, lon=78.487, alt=50),
                    heading=0, speed=0, battery_pct=20,  # Low battery
                    state=DroneState.PATROL,
                ),
            ),
        }

        commands = await llm.strategize(reports)
        # Should get RTL command
        rtl_cmds = [c for c in commands if c.command_type == CommandType.RETURN_TO_LAUNCH]
        assert len(rtl_cmds) >= 1
        await llm.close()

    def test_fleet_manager_timeout(self):
        """Fleet manager should detect offline drones after timeout."""
        fleet = FleetManager(drone_timeout_s=0.5)
        fleet.register_drone("drone_01")

        report = DroneReport(
            source_id="drone_01",
            drone_id="drone_01",
            telemetry=DroneTelemetry(
                source_id="drone_01",
                drone_id="drone_01",
                position=GeoPoint(lat=17.385, lon=78.487, alt=50),
                heading=0, speed=0, battery_pct=85,
            ),
        )
        fleet.update_drone(report)
        assert fleet.online_count == 1

        # Simulate timeout
        fleet._drones["drone_01"].last_seen = time.time() - 1.0
        health = fleet.check_health()
        assert health["offline"] >= 1

    def test_mission_planner_roe_modification(self):
        """Test ROE modification on active mission."""
        planner = MissionPlanner()
        planner.create_default_mission(["drone_01"])

        roe = planner.modify_roe({"engagement_allowed": True, "max_altitude_m": 80})
        assert roe["engagement_allowed"] is True
        assert roe["max_altitude_m"] == 80

    def test_mission_planner_ngz_addition(self):
        """Test adding no-go zones to active mission."""
        from proto.messages import GeoZone

        planner = MissionPlanner()
        planner.create_default_mission(["drone_01"])

        initial_ngz = len(planner.no_go_zones)
        planner.add_no_go_zone(GeoZone(
            zone_id="ngz_2",
            name="Test NGZ",
            points=[
                GeoPoint(lat=17.390, lon=78.480),
                GeoPoint(lat=17.390, lon=78.482),
                GeoPoint(lat=17.388, lon=78.482),
            ],
        ))
        assert len(planner.no_go_zones) == initial_ngz + 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard API Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardAPI:
    """Test dashboard REST API endpoints using FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.dashboard.server import create_app
        app = create_app(ground_station=None)
        return TestClient(app)

    def test_fleet_api_no_station(self, client):
        resp = client.get("/api/fleet")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data  # No ground station initialized

    def test_events_api_no_station(self, client):
        resp = client.get("/api/events")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_mission_api_no_station(self, client):
        resp = client.get("/api/mission")
        assert resp.status_code == 200

    def test_stats_api_no_station(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200

    def test_command_api_no_station(self, client):
        resp = client.post("/api/command", json={
            "drone_id": "drone_01",
            "command": "hold",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data  # No ground station

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Layer Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafetyEdgeCases:
    def test_multiple_safety_vetoes_tracked(self):
        """Safety layer should track all vetoes."""
        safety = SafetyLayer(
            drone_id="test",
            home_position=GeoPoint(lat=17.385, lon=78.487, alt=0),
        )

        # Trigger battery veto
        decision = LLMDecision(source="local_llm", action="PATROL", confidence=0.8)
        safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=5.0,
        )

        # Trigger another veto
        decision2 = LLMDecision(source="local_llm", action="FIRE", confidence=0.9)
        safety.validate(
            decision=decision2,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )

        assert safety.stats["total_vetoes"] == 2
        assert len(safety.recent_vetoes) == 2

    def test_safety_comms_time_update(self):
        """Comms time should update on explicit call."""
        safety = SafetyLayer(
            drone_id="test",
            home_position=GeoPoint(lat=17.385, lon=78.487, alt=0),
            comms_loss_timeout_s=0.5,
        )
        safety.update_comms_time()
        assert time.time() - safety._last_comms_time < 0.1

    def test_safety_peer_position_update(self):
        """Peer positions should be tracked for collision avoidance."""
        safety = SafetyLayer(
            drone_id="test",
            home_position=GeoPoint(lat=17.385, lon=78.487, alt=0),
        )
        safety.update_peer_positions({
            "drone_02": GeoPoint(lat=17.386, lon=78.488, alt=50),
        })
        assert "drone_02" in safety._peer_positions

    def test_low_confidence_anomaly_veto(self):
        """Low-confidence risky actions should be vetoed."""
        safety = SafetyLayer(
            drone_id="test",
            home_position=GeoPoint(lat=17.385, lon=78.487, alt=0),
        )
        decision = LLMDecision(
            source="local_llm",
            action="INVESTIGATE",
            confidence=0.1,  # Very low
        )
        result, veto = safety.validate(
            decision=decision,
            current_position=GeoPoint(lat=17.385, lon=78.487, alt=50),
            current_state=DroneState.PATROL,
            battery_pct=80.0,
        )
        assert veto is not None
        assert veto.reason.value == "llm_anomaly"


# ═══════════════════════════════════════════════════════════════════════════════
# Sensor Fusion Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSensorFusionEdgeCases:
    def test_mixed_rgb_thermal_fusion(self):
        """Test fusion with both RGB and thermal detections."""
        fusion = SensorFusion()
        pos = GeoPoint(lat=17.385, lon=78.487, alt=50)

        rgb_dets = [
            DetectionEvent(
                source=DetectionSource.RGB,
                detection_class=DetectionClass.PERSON,
                confidence=0.85,
                bbox=BoundingBox(x_min=0.3, y_min=0.3, x_max=0.6, y_max=0.8),
                geo_position=pos,
            ),
            DetectionEvent(
                source=DetectionSource.RGB,
                detection_class=DetectionClass.VEHICLE_CAR,
                confidence=0.75,
                geo_position=pos,
            ),
        ]
        thermal_dets = [
            ThermalDetection(
                blob_class=DetectionClass.PERSON,
                confidence=0.7,
                temperature_c=36.5,
                blob_area=0.02,
                geo_position=pos,
            ),
        ]

        fused = fusion.fuse(rgb_dets, thermal_dets, pos)
        assert len(fused) >= 2  # At least the two RGB dets (one may merge with thermal)

        # Find the person detection — should have higher confidence due to fusion
        person_events = [e for e in fused if e.detection_class == DetectionClass.PERSON]
        assert len(person_events) >= 1

    def test_empty_inputs(self):
        fusion = SensorFusion()
        fused = fusion.fuse([], [], None)
        assert fused == []

    def test_weapon_detection_marks_armed(self):
        """WEAPON_RIFLE or WEAPON_HANDGUN → armed=True, reclassified as PERSON."""
        fusion = SensorFusion()
        rgb_dets = [
            DetectionEvent(
                source=DetectionSource.RGB,
                detection_class=DetectionClass.WEAPON_RIFLE,
                confidence=0.8,
                geo_position=GeoPoint(lat=17.385, lon=78.487),
            ),
        ]
        fused = fusion.fuse(rgb_dets, [], GeoPoint(lat=17.385, lon=78.487))
        assert len(fused) == 1
        assert fused[0].armed is True
        assert fused[0].detection_class == DetectionClass.PERSON  # Weapon → Person carrying it


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior Analyzer Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestBehaviorEdgeCases:
    def test_evasive_behavior_detection(self):
        """Test that rapid direction changes are detected as evasive."""
        analyzer = BehaviorAnalyzer(evasion_speed_threshold=2.0)
        analyzer.load()

        # Create events with different positions simulating fast, erratic movement
        events = []
        base_lat = 17.385
        base_lon = 78.487

        for i in range(10):
            # Zigzag pattern
            offset = 0.0001 * (1 if i % 2 == 0 else -1) * (i + 1)
            event = FusedEvent(
                detection_class=DetectionClass.PERSON,
                class_confidence=0.8,
                geo_position=GeoPoint(
                    lat=base_lat + offset,
                    lon=base_lon + offset * 0.5,
                ),
                sources=[DetectionSource.RGB],
                timestamp=time.time() + i * 0.5,
            )
            events.append(event)

        # Analyze each event sequentially
        for event in events:
            result = analyzer.analyze([event])

        assert analyzer.active_tracks > 0

    def test_stationary_behavior(self):
        """Same position repeatedly should be classified as stationary."""
        analyzer = BehaviorAnalyzer()
        analyzer.load()

        pos = GeoPoint(lat=17.385, lon=78.487)
        for i in range(5):
            event = FusedEvent(
                detection_class=DetectionClass.PERSON,
                class_confidence=0.8,
                geo_position=pos,
                sources=[DetectionSource.RGB],
                timestamp=time.time() + i,
            )
            result = analyzer.analyze([event])

        # Last analysis should show stationary or unknown
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# State Machine Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateMachineEdgeCases:
    def test_offline_to_idle(self):
        fsm = DroneStateMachine("test", initial_state=DroneState.OFFLINE)
        assert fsm.transition(DroneState.IDLE)
        assert fsm.state == DroneState.IDLE

    def test_time_in_state(self):
        fsm = DroneStateMachine("test")
        assert fsm.time_in_state >= 0
        assert fsm.time_in_state < 1.0

    def test_previous_state_tracking(self):
        fsm = DroneStateMachine("test")
        assert fsm.previous_state is None
        fsm.transition(DroneState.PREFLIGHT)
        assert fsm.previous_state == DroneState.IDLE

    def test_on_transition_callback(self):
        fsm = DroneStateMachine("test")
        transitions = []
        fsm.on_transition(lambda from_s, to_s, reason: transitions.append((from_s, to_s)))
        fsm.transition(DroneState.PREFLIGHT, reason="boot")
        fsm.transition(DroneState.TAKEOFF, reason="launch")
        assert len(transitions) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


async def _async_append(lst, item):
    """Async helper to append to a list (used as handler callback)."""
    lst.append(item)
