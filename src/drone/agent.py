"""
VayuSwarm — Drone Agent Orchestrator

The main drone agent that runs the complete pipeline:
  Camera → YOLO → Thermal → Fusion → Behavior → LLM → Safety → PX4

Manages drone lifecycle, handles ground station commands, and coordinates
with peer drones via mesh network.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import numpy as np
import structlog
import yaml

from proto.messages import (
    CommandType,
    DroneReport,
    DroneState,
    DroneTelemetry,
    FusedEvent,
    GeoPoint,
    GroundCommand,
    LLMDecision,
    SwarmMessage,
)
from src.comms.message_bus import MessageBus
from src.comms.mesh import MeshNetwork
from src.drone.local_llm import LocalLLM
from src.drone.px4_interface import PX4Interface
from src.drone.safety_layer import SafetyLayer
from src.drone.state_machine import DroneStateMachine
from src.vision.behavior_analyzer import BehaviorAnalyzer
from src.vision.sensor_fusion import SensorFusion
from src.vision.thermal_model import ThermalModel
from src.vision.yolo_detector import YOLODetector

logger = structlog.get_logger(__name__)


class DroneAgent:
    """
    Main drone agent — orchestrates the full perception-to-action pipeline.
    
    Pipeline:
    1. Capture frames (RGB + Thermal)
    2. YOLOv8 detection on RGB
    3. Thermal blob analysis
    4. Sensor fusion (merge RGB + thermal)
    5. Behavior analysis (temporal Transformer)
    6. Local LLM decision
    7. Safety layer validation
    8. PX4 command execution
    9. Report to ground station
    """

    def __init__(
        self,
        drone_id: str,
        config: Optional[dict] = None,
    ):
        self._drone_id = drone_id
        self._config = config or {}
        self._running = False

        # ── Load config ──
        sys_cfg = self._config.get("system", {})
        comms_cfg = self._config.get("comms", {})
        vision_cfg = self._config.get("vision", {})
        llm_cfg = self._config.get("llm", {}).get("drone", {})
        safety_cfg = self._config.get("safety", {})
        drone_cfg = self._config.get("drone", {})
        
        simulation = sys_cfg.get("simulation", True)

        # ── Home position ──
        home_cfg = drone_cfg.get("home", {})
        self._home = GeoPoint(
            lat=home_cfg.get("lat", 17.385044),
            lon=home_cfg.get("lon", 78.486671),
            alt=home_cfg.get("alt", 0),
        )

        # ── Vision Pipeline ──
        yolo_cfg = vision_cfg.get("yolo", {})
        self.yolo = YOLODetector(
            model_path=yolo_cfg.get("model_path", "yolov8n.pt"),
            confidence_threshold=yolo_cfg.get("confidence_threshold", 0.5),
            iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
            device=yolo_cfg.get("device", "auto"),
            class_thresholds=yolo_cfg.get("classes", {}),
        )

        thermal_cfg = vision_cfg.get("thermal", {})
        self.thermal = ThermalModel(
            model_path=thermal_cfg.get("model_path"),
        )

        fusion_cfg = vision_cfg.get("fusion", {})
        self.fusion = SensorFusion(
            iou_match_threshold=fusion_cfg.get("iou_match_threshold", 0.3),
            confidence_boost=fusion_cfg.get("confidence_boost", 0.15),
        )

        behavior_cfg = vision_cfg.get("behavior", {})
        self.behavior = BehaviorAnalyzer(
            window_size=behavior_cfg.get("window_size", 30),
            model_path=behavior_cfg.get("model_path"),
            evasion_speed_threshold=behavior_cfg.get("evasion_speed_threshold_ms", 3.0),
        )

        # ── Local LLM ──
        self.llm = LocalLLM(
            drone_id=drone_id,
            provider=llm_cfg.get("provider", "ollama"),
            model=llm_cfg.get("model", "phi3:3.8b"),
            endpoint=llm_cfg.get("endpoint", "http://localhost:11434"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 512),
            timeout_s=llm_cfg.get("timeout_s", 5.0),
        )

        # ── Safety Layer ──
        self.safety = SafetyLayer(
            drone_id=drone_id,
            home_position=self._home,
            max_altitude_m=safety_cfg.get("max_altitude_m", 120.0),
            min_altitude_m=safety_cfg.get("min_altitude_m", 5.0),
            battery_critical_pct=safety_cfg.get("battery_critical_pct", 15.0),
            collision_radius_m=safety_cfg.get("collision_radius_m", 10.0),
            comms_loss_timeout_s=safety_cfg.get("comms_loss_timeout_s", 15.0),
            geofence_radius_m=safety_cfg.get("geofence_radius_m", 5000.0),
        )

        # ── PX4 Interface ──
        px4_cfg = drone_cfg.get("px4", {})
        self.px4 = PX4Interface(
            drone_id=drone_id,
            connection_string=px4_cfg.get("connection_string", "udp:127.0.0.1:14540"),
            simulation=simulation,
        )

        # ── State Machine ──
        self.fsm = DroneStateMachine(drone_id=drone_id)

        # ── Communication ──
        self.message_bus: Optional[MessageBus] = None
        self.mesh: Optional[MeshNetwork] = None

        # ── Pipeline state ──
        self._current_events: list[FusedEvent] = []
        self._last_decision: Optional[LLMDecision] = None
        self._last_report_time = 0.0
        self._report_interval = 2.0  # seconds
        self._pipeline_interval = 0.1  # seconds (10 Hz)
        self._cycle_count = 0

        # ── Robustness / Degradation tracking ──
        self._last_known_position: Optional[GeoPoint] = None
        self._gps_failures = 0
        self._camera_failures = 0
        self._llm_failures = 0
        self._camera_max_failures = 5    # switch to mock after N failures
        self._llm_max_failures = 3       # fallback to HOLD after N failures
        self._use_mock_camera = False

        # ── Mission context ──
        self._mission_description = "Surveillance patrol"
        self._roe: dict = {}

    async def start(self, ground_address: str = "tcp://localhost:5555") -> None:
        """Initialize all subsystems and start the main loop."""
        logger.info("drone_agent.starting", drone_id=self._drone_id)

        # Load models
        self.yolo.load()
        self.thermal.load()
        self.behavior.load()

        # Connect PX4
        await self.px4.connect()

        # Connect LLM
        await self.llm.connect()

        # Set up message bus
        comms_cfg = self._config.get("comms", {})
        sub_port = comms_cfg.get("ground_pub_port", 5555)
        pub_port = comms_cfg.get("ground_sub_port", 5556)

        self.message_bus = MessageBus(
            role="drone",
            pub_bind_addr=f"tcp://*:{pub_port + int(self._drone_id.split('_')[-1]) if '_' in self._drone_id else pub_port}",
            sub_connect_addr=ground_address,
            node_id=self._drone_id,
            topics=[f"ground/commands/{self._drone_id}", "ground/commands/all", "heartbeat/"],
        )

        # Register handlers
        self.message_bus.on_message(f"ground/commands/{self._drone_id}", self._handle_ground_command)
        self.message_bus.on_message("ground/commands/all", self._handle_ground_command)

        await self.message_bus.start()

        # Set up mesh network
        mesh_port = comms_cfg.get("mesh_base_port", 6000)
        drone_num = int(self._drone_id.split("_")[-1]) if "_" in self._drone_id else 0
        self.mesh = MeshNetwork(
            drone_id=self._drone_id,
            bind_port=mesh_port + drone_num,
        )
        self.mesh.on_message(self._handle_swarm_message)
        await self.mesh.start()

        self._running = True
        logger.info("drone_agent.started", drone_id=self._drone_id)

        # Start main loop
        await self._main_loop()

    async def _main_loop(self) -> None:
        """Main perception-to-action loop with robustness hardening."""
        while self._running:
            try:
                cycle_start = time.time()
                self._cycle_count += 1

                # Step 1: Get telemetry (GPS failure handling)
                try:
                    telemetry = await self.px4.get_telemetry()
                    if telemetry.position:
                        self._last_known_position = telemetry.position
                        self._gps_failures = 0
                    else:
                        raise ValueError("No GPS fix")
                except Exception as e:
                    self._gps_failures += 1
                    if self._gps_failures <= 3:
                        logger.warning("drone_agent.gps_degraded",
                                       failures=self._gps_failures,
                                       using="last_known_position")
                    if self._last_known_position is None:
                        logger.error("drone_agent.no_position_available")
                        await asyncio.sleep(1.0)
                        continue
                    # Use last known position
                    telemetry = await self.px4.get_telemetry()

                drone_pos = self._last_known_position or telemetry.position

                # Step 2: Capture frames (camera failure handling)
                try:
                    if self._use_mock_camera:
                        rgb_frame = self._capture_rgb()
                        thermal_frame = self._capture_thermal()
                    else:
                        rgb_frame = self._capture_rgb()
                        thermal_frame = self._capture_thermal()
                        self._camera_failures = 0
                except Exception as e:
                    self._camera_failures += 1
                    logger.warning("drone_agent.camera_failure",
                                   failures=self._camera_failures,
                                   error=str(e))
                    if self._camera_failures >= self._camera_max_failures:
                        if not self._use_mock_camera:
                            logger.warning("drone_agent.switching_to_mock_camera")
                            self._use_mock_camera = True
                    rgb_frame = self._capture_rgb()  # fallback to simulated
                    thermal_frame = self._capture_thermal()

                # Step 3: YOLOv8 detection
                try:
                    rgb_detections = self.yolo.detect(rgb_frame, drone_pos)
                except Exception as e:
                    logger.error("drone_agent.yolo_error", error=str(e))
                    rgb_detections = []

                # Step 4: Thermal analysis
                try:
                    if thermal_frame is not None:
                        thermal_detections = self.thermal.detect(thermal_frame, drone_pos)
                    else:
                        thermal_detections = self.thermal.mock_detect(drone_pos)
                except Exception as e:
                    logger.error("drone_agent.thermal_error", error=str(e))
                    thermal_detections = []

                # Step 5: Sensor fusion
                fused_events = self.fusion.fuse(
                    rgb_detections, thermal_detections, drone_pos
                )

                # Step 6: Behavior analysis
                if fused_events:
                    try:
                        fused_events = self.behavior.analyze(fused_events)
                    except Exception as e:
                        logger.error("drone_agent.behavior_error", error=str(e))

                self._current_events = fused_events

                # Step 7: LLM decision (with failure fallback)
                if fused_events or self._cycle_count % 50 == 0:
                    try:
                        decision = await self.llm.decide(
                            fused_events=fused_events,
                            drone_state=self.fsm.state,
                            drone_position=drone_pos,
                            battery_pct=telemetry.battery_pct,
                            mission_description=self._mission_description,
                            roe=self._roe,
                        )
                        self._llm_failures = 0
                    except Exception as e:
                        self._llm_failures += 1
                        logger.warning("drone_agent.llm_failure",
                                       failures=self._llm_failures,
                                       error=str(e))
                        # Fallback: HOLD position if LLM is repeatedly failing
                        if self._llm_failures >= self._llm_max_failures:
                            logger.warning("drone_agent.llm_fallback_hold")
                            decision = LLMDecision(
                                source="safety_fallback",
                                action="HOLD",
                                reasoning=f"LLM unavailable ({self._llm_failures} failures), holding position",
                                confidence=1.0,
                            )
                        else:
                            decision = LLMDecision(
                                source="safety_fallback",
                                action="CONTINUE",
                                reasoning="LLM timeout, continuing current behavior",
                                confidence=0.5,
                            )

                    # Step 8: Safety validation
                    safe_decision, veto = self.safety.validate(
                        decision=decision,
                        current_position=drone_pos,
                        current_state=self.fsm.state,
                        battery_pct=telemetry.battery_pct,
                    )

                    self._last_decision = safe_decision

                    # Step 9: Execute decision
                    await self._execute_decision(safe_decision)

                # Step 10: Report to ground station
                now = time.time()
                if now - self._last_report_time >= self._report_interval:
                    try:
                        await self._send_report(telemetry)
                    except Exception as e:
                        logger.warning("drone_agent.report_failed", error=str(e))
                    self._last_report_time = now

                # Step 11: Share position with swarm
                if self.mesh and self._cycle_count % 10 == 0:
                    try:
                        swarm_msg = SwarmMessage(
                            source_id=self._drone_id,
                            sender_drone_id=self._drone_id,
                            position=drone_pos,
                            heading=telemetry.heading,
                            speed=telemetry.speed,
                        )
                        await self.mesh.broadcast(swarm_msg)
                    except Exception as e:
                        logger.warning("drone_agent.mesh_send_failed", error=str(e))

                # Maintain loop rate
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._pipeline_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("drone_agent.loop_error", error=str(e), cycle=self._cycle_count)
                await asyncio.sleep(1.0)

    async def _execute_decision(self, decision: LLMDecision) -> None:
        """Execute an LLM decision via PX4."""
        action = decision.action.upper()

        # Try FSM transition
        target_state = self.fsm.action_to_state(action)
        if target_state and target_state != self.fsm.state:
            self.fsm.transition(target_state, reason=decision.reasoning[:100])

        # Execute PX4 commands
        if action == "RTL":
            await self.px4.return_to_launch()
        elif action == "LAND":
            await self.px4.land()
        elif action == "HOLD":
            await self.px4.hold_position()
        elif action in ("INVESTIGATE", "TRACK") and decision.suggested_waypoint:
            await self.px4.goto(decision.suggested_waypoint)
        elif action == "AVOID" and decision.suggested_waypoint:
            await self.px4.goto(decision.suggested_waypoint, speed_ms=15.0)
        elif action == "CONTINUE":
            pass  # Keep current behavior

    async def _send_report(self, telemetry: DroneTelemetry) -> None:
        """Send aggregated report to ground station."""
        if not self.message_bus:
            return

        report = DroneReport(
            source_id=self._drone_id,
            drone_id=self._drone_id,
            telemetry=telemetry,
            fused_events=self._current_events,
            local_llm_decision=self._last_decision.action if self._last_decision else None,
            safety_vetoes=[v.reason.value for v in self.safety.recent_vetoes[-3:]],
            current_action=self._last_decision.action if self._last_decision else "idle",
            notes=self._last_decision.reasoning[:200] if self._last_decision else "",
        )

        await self.message_bus.publish(f"drone/{self._drone_id}/report", report)

    async def _handle_ground_command(self, topic: str, msg) -> None:
        """Handle a command from the ground station."""
        if not isinstance(msg, GroundCommand):
            return

        self.safety.update_comms_time()
        logger.info("drone_agent.ground_command",
                     drone_id=self._drone_id,
                     command=msg.command_type.value,
                     priority=msg.priority.value)

        if msg.command_type == CommandType.GOTO_WAYPOINT and msg.waypoint:
            await self.px4.goto(msg.waypoint)
            self.fsm.transition(DroneState.PATROL, reason="Ground command: goto_waypoint")

        elif msg.command_type == CommandType.RETURN_TO_LAUNCH:
            await self.px4.return_to_launch()
            self.fsm.transition(DroneState.RTL, reason="Ground command: RTL")

        elif msg.command_type == CommandType.LAND:
            await self.px4.land()
            self.fsm.transition(DroneState.LAND, reason="Ground command: land")

        elif msg.command_type == CommandType.HOLD_POSITION:
            await self.px4.hold_position()

        elif msg.command_type == CommandType.INVESTIGATE_TARGET and msg.waypoint:
            await self.px4.goto(msg.waypoint)
            self.fsm.transition(DroneState.INVESTIGATE, reason="Ground command: investigate")

        elif msg.command_type == CommandType.TRACK_TARGET and msg.waypoint:
            await self.px4.goto(msg.waypoint)
            self.fsm.transition(DroneState.TRACK, reason="Ground command: track_target")

        elif msg.command_type == CommandType.CHANGE_ALTITUDE and msg.altitude:
            pos = (await self.px4.get_telemetry()).position
            await self.px4.goto(GeoPoint(lat=pos.lat, lon=pos.lon, alt=msg.altitude))

        elif msg.command_type == CommandType.UPDATE_ROE and msg.roe_update:
            self._roe.update(msg.roe_update)
            logger.info("drone_agent.roe_updated", drone_id=self._drone_id)

        elif msg.command_type == CommandType.EMERGENCY_STOP:
            await self.px4.hold_position()
            self.fsm.transition(DroneState.EMERGENCY, reason="Ground command: emergency_stop")

        elif msg.command_type == CommandType.SET_PATROL_ZONE and msg.patrol_zone:
            self._mission_description = f"Patrolling zone: {msg.patrol_zone.name}"

    async def _handle_swarm_message(self, sender_id: str, msg) -> None:
        """Handle a message from a peer drone."""
        if isinstance(msg, SwarmMessage):
            # Update peer positions for collision avoidance
            if msg.position:
                self.safety.update_peer_positions({sender_id: msg.position})

            # Handle target handoff
            if msg.handoff_event:
                logger.info("drone_agent.target_handoff",
                            from_drone=sender_id,
                            event_class=msg.handoff_event.detection_class.value)
                self._current_events.append(msg.handoff_event)

    def _capture_rgb(self) -> np.ndarray:
        """Capture RGB frame (simulated — returns random noise)."""
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    def _capture_thermal(self) -> Optional[np.ndarray]:
        """Capture thermal frame (simulated — returns random thermal data)."""
        return np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    async def stop(self) -> None:
        """Shutdown all subsystems."""
        self._running = False
        logger.info("drone_agent.stopping", drone_id=self._drone_id)

        if self.mesh:
            await self.mesh.stop()
        if self.message_bus:
            await self.message_bus.stop()
        await self.llm.close()
        await self.px4.disconnect()

        logger.info("drone_agent.stopped", drone_id=self._drone_id)

    @property
    def stats(self) -> dict:
        return {
            "drone_id": self._drone_id,
            "state": self.fsm.state.value,
            "cycles": self._cycle_count,
            "active_events": len(self._current_events),
            "yolo": self.yolo.stats,
            "thermal": self.thermal.stats,
            "behavior": self.behavior.stats,
            "llm": self.llm.stats,
            "safety": self.safety.stats,
        }
