"""
VayuSwarm — Ground Station Orchestrator

Main ground station that receives reports from all drones,
runs the strategic 70B LLM, and dispatches commands.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog
import yaml

from proto.messages import (
    DroneReport,
    GroundCommand,
    MissionDefinition,
)
from src.comms.message_bus import MessageBus
from src.ground.fleet_manager import FleetManager
from src.ground.ground_llm import GroundLLM
from src.ground.mission_planner import MissionPlanner

logger = structlog.get_logger(__name__)


class GroundStation:
    """
    Ground Station Orchestrator.
    
    - Receives reports from all drones via message bus
    - Aggregates intelligence
    - Runs strategic 70B LLM on aggregated data
    - Dispatches commands to individual drones
    - Manages mission lifecycle
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._running = False

        # ── Config ──
        llm_cfg = self._config.get("llm", {}).get("ground", {})
        comms_cfg = self._config.get("comms", {})
        gs_cfg = self._config.get("ground_station", {})
        decision_cfg = gs_cfg.get("decision", {})

        # ── Components ──
        self.fleet = FleetManager(
            drone_timeout_s=gs_cfg.get("fleet", {}).get("drone_timeout_s", 15.0),
        )

        self.mission_planner = MissionPlanner()

        self.llm = GroundLLM(
            provider=llm_cfg.get("provider", "ollama"),
            model=llm_cfg.get("model", "llama3:70b"),
            endpoint=llm_cfg.get("endpoint", "http://localhost:11434"),
            temperature=llm_cfg.get("temperature", 0.3),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            timeout_s=llm_cfg.get("timeout_s", 30.0),
        )

        self.message_bus: Optional[MessageBus] = None

        # ── Decision cycle ──
        self._decision_interval = decision_cfg.get("cycle_interval_s", 5.0)
        self._min_reports = decision_cfg.get("min_reports_for_decision", 1)
        self._cycle_count = 0
        self._last_decision_time = 0.0

        # ── Dashboard data ──
        self._event_log: list[dict] = []
        self._command_log: list[dict] = []

    async def start(self, drone_ids: Optional[list[str]] = None) -> None:
        """Start the ground station."""
        logger.info("ground_station.starting")

        # Register drones
        drone_ids = drone_ids or ["drone_01", "drone_02", "drone_03"]
        for drone_id in drone_ids:
            self.fleet.register_drone(drone_id)

        # Create default mission
        mission = self.mission_planner.create_default_mission(drone_ids)

        # Connect LLM
        await self.llm.connect()

        # Set up message bus
        comms_cfg = self._config.get("comms", {})
        pub_port = comms_cfg.get("ground_pub_port", 5555)
        sub_port = comms_cfg.get("ground_sub_port", 5556)

        self.message_bus = MessageBus(
            role="ground",
            pub_addr=f"tcp://*:{pub_port}",
            sub_addr=f"tcp://*:{sub_port}",
            node_id="ground",
            topics=["drone/", "heartbeat/"],
            pub_bind=True,
            sub_bind=True,
        )

        # Register handlers
        self.message_bus.on_message("drone/", self._handle_drone_report)
        self.message_bus.on_message("heartbeat/", self._handle_heartbeat)
        await self.message_bus.start()

        self._running = True
        logger.info("ground_station.started", drones=len(drone_ids))

        # Start main loop
        await self._main_loop()

    async def _main_loop(self) -> None:
        """Main ground station decision loop."""
        while self._running:
            try:
                now = time.time()
                self._cycle_count += 1

                # Check fleet health
                health = self.fleet.check_health()
                if health["issues"]:
                    logger.info("ground_station.health_check", issues=health["issues"])

                # Strategic decision cycle
                if now - self._last_decision_time >= self._decision_interval:
                    reports = self.fleet.get_all_reports()

                    if len(reports) >= self._min_reports:
                        # Run strategic LLM
                        commands = await self.llm.strategize(
                            drone_reports=reports,
                            mission=self.mission_planner.active_mission,
                        )

                        # Dispatch commands
                        for cmd in commands:
                            await self._dispatch_command(cmd)

                        self._last_decision_time = now

                    elif self._cycle_count % 10 == 0:
                        logger.debug("ground_station.waiting_for_reports",
                                     have=len(reports),
                                     need=self._min_reports)

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ground_station.loop_error", error=str(e))
                await asyncio.sleep(2.0)

    async def _handle_heartbeat(self, topic: str, msg) -> None:
        """Handle heartbeat from drones."""
        # Extract drone_id from topic like 'heartbeat/drone_01'
        parts = topic.split("/")
        if len(parts) >= 2:
            node_id = parts[1]
            self.message_bus.heartbeat.beat(node_id)

    async def _handle_drone_report(self, topic: str, msg) -> None:
        """Handle incoming drone report."""
        if not isinstance(msg, DroneReport):
            return

        self.fleet.update_drone(msg)

        # Log events for dashboard
        if msg.fused_events:
            for event in msg.fused_events:
                self._event_log.append({
                    "timestamp": time.time(),
                    "drone_id": msg.drone_id,
                    "event": event.to_llm_text(),
                    "threat_level": event.threat_level.value,
                })

        # Keep log bounded
        if len(self._event_log) > 500:
            self._event_log = self._event_log[-500:]

        logger.debug("ground_station.report_received",
                      drone_id=msg.drone_id,
                      events=len(msg.fused_events))

    async def _dispatch_command(self, cmd: GroundCommand) -> None:
        """Send a command to a specific drone."""
        if not self.message_bus:
            return

        topic = f"ground/commands/{cmd.target_drone_id}"
        await self.message_bus.publish(topic, cmd)

        # Log for dashboard
        self._command_log.append({
            "timestamp": time.time(),
            "drone_id": cmd.target_drone_id,
            "command": cmd.command_type.value,
            "priority": cmd.priority.value,
            "message": cmd.message[:100],
        })

        if len(self._command_log) > 200:
            self._command_log = self._command_log[-200:]

        logger.info("ground_station.command_sent",
                     drone_id=cmd.target_drone_id,
                     command=cmd.command_type.value,
                     priority=cmd.priority.value)

    async def send_command(self, cmd: GroundCommand) -> None:
        """Public method to send a command (for dashboard API)."""
        await self._dispatch_command(cmd)

    @property
    def event_log(self) -> list[dict]:
        return self._event_log[-50:]

    @property
    def command_log(self) -> list[dict]:
        return self._command_log[-50:]

    @property
    def stats(self) -> dict:
        return {
            "cycles": self._cycle_count,
            "fleet": {
                "total": self.fleet.drone_count,
                "online": self.fleet.online_count,
            },
            "mission": self.mission_planner.progress,
            "llm": self.llm.stats,
            "events_logged": len(self._event_log),
            "commands_sent": len(self._command_log),
        }

    async def stop(self) -> None:
        """Stop the ground station."""
        self._running = False
        if self.message_bus:
            await self.message_bus.stop()
        await self.llm.close()
        logger.info("ground_station.stopped")
