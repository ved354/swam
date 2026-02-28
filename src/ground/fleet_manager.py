"""
VayuSwarm — Fleet Manager

Tracks all drones in the swarm: status, health, telemetry.
Handles automatic reassignment on drone loss and formation management.
"""

from __future__ import annotations

import time
from typing import Optional

import structlog

from proto.messages import (
    DroneReport,
    DroneState,
    DroneTelemetry,
    GeoPoint,
)

logger = structlog.get_logger(__name__)


class DroneStatus:
    """Maintains the current status of a single drone."""

    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        self.last_report: Optional[DroneReport] = None
        self.last_telemetry: Optional[DroneTelemetry] = None
        self.last_seen: float = 0.0
        self.online: bool = False
        self.warnings: list[str] = []

    def update(self, report: DroneReport) -> None:
        """Update status from a drone report."""
        self.last_report = report
        self.last_telemetry = report.telemetry
        self.last_seen = time.time()
        self.online = True

        # Check for warnings
        self.warnings.clear()
        if report.telemetry.battery_pct < 25:
            self.warnings.append(f"LOW_BATTERY ({report.telemetry.battery_pct:.0f}%)")
        if report.telemetry.signal_strength < 50:
            self.warnings.append(f"WEAK_SIGNAL ({report.telemetry.signal_strength:.0f}%)")
        if report.telemetry.gps_fix < 3:
            self.warnings.append(f"GPS_DEGRADED (fix={report.telemetry.gps_fix})")
        if report.safety_vetoes:
            self.warnings.append(f"SAFETY_VETOES: {', '.join(report.safety_vetoes)}")

    @property
    def position(self) -> Optional[GeoPoint]:
        return self.last_telemetry.position if self.last_telemetry else None

    @property
    def state(self) -> DroneState:
        if self.last_telemetry:
            return self.last_telemetry.state
        return DroneState.OFFLINE

    @property
    def battery(self) -> float:
        return self.last_telemetry.battery_pct if self.last_telemetry else 0

    def to_dict(self) -> dict:
        """Serialize to dict for dashboard."""
        return {
            "drone_id": self.drone_id,
            "online": self.online,
            "state": self.state.value,
            "battery": self.battery,
            "position": {
                "lat": self.position.lat,
                "lon": self.position.lon,
                "alt": self.position.alt,
            } if self.position else None,
            "heading": self.last_telemetry.heading if self.last_telemetry else 0,
            "speed": self.last_telemetry.speed if self.last_telemetry else 0,
            "signal": self.last_telemetry.signal_strength if self.last_telemetry else 0,
            "warnings": self.warnings,
            "last_action": self.last_report.current_action if self.last_report else "unknown",
            "events_count": len(self.last_report.fused_events) if self.last_report else 0,
            "last_seen": self.last_seen,
        }


class FleetManager:
    """
    Manages the drone fleet.
    
    Responsibilities:
    - Track real-time status of all drones
    - Health monitoring (battery, signal, GPS)
    - Detect offline drones
    - Provide aggregated fleet status for ground LLM
    """

    def __init__(self, drone_timeout_s: float = 15.0):
        self._drones: dict[str, DroneStatus] = {}
        self._drone_timeout = drone_timeout_s
        self._lost_drones: set[str] = set()
        self._on_drone_lost_callbacks: list = []

    def register_drone(self, drone_id: str) -> None:
        """Register a drone in the fleet."""
        self._drones[drone_id] = DroneStatus(drone_id)
        logger.info("fleet.drone_registered", drone_id=drone_id)

    def update_drone(self, report: DroneReport) -> None:
        """Update a drone's status from its report."""
        drone_id = report.drone_id
        if drone_id not in self._drones:
            self.register_drone(drone_id)

        self._drones[drone_id].update(report)

        # If it was lost, mark as recovered
        if drone_id in self._lost_drones:
            self._lost_drones.discard(drone_id)
            logger.info("fleet.drone_recovered", drone_id=drone_id)

    def check_health(self) -> dict:
        """
        Check fleet health. Returns status summary and list of issues.
        """
        now = time.time()
        issues = []
        online = 0
        offline = 0

        for drone_id, status in self._drones.items():
            if status.online and (now - status.last_seen > self._drone_timeout):
                status.online = False
                self._lost_drones.add(drone_id)
                issues.append(f"{drone_id}: OFFLINE (timeout)")
                logger.warning("fleet.drone_lost", drone_id=drone_id)
                offline += 1
            elif status.online:
                online += 1
                if status.warnings:
                    issues.append(f"{drone_id}: {', '.join(status.warnings)}")
            else:
                offline += 1

        return {
            "total": len(self._drones),
            "online": online,
            "offline": offline,
            "lost": list(self._lost_drones),
            "issues": issues,
        }

    def get_all_reports(self) -> dict[str, DroneReport]:
        """Get latest reports from all online drones."""
        reports = {}
        for drone_id, status in self._drones.items():
            if status.online and status.last_report:
                reports[drone_id] = status.last_report
        return reports

    def get_all_positions(self) -> dict[str, GeoPoint]:
        """Get current positions of all online drones."""
        positions = {}
        for drone_id, status in self._drones.items():
            if status.online and status.position:
                positions[drone_id] = status.position
        return positions

    def get_drone_status(self, drone_id: str) -> Optional[DroneStatus]:
        """Get status of a specific drone."""
        return self._drones.get(drone_id)

    def get_fleet_summary(self) -> list[dict]:
        """Get fleet summary for dashboard."""
        return [status.to_dict() for status in self._drones.values()]

    def get_online_drone_ids(self) -> list[str]:
        """Get list of online drone IDs."""
        return [did for did, s in self._drones.items() if s.online]

    @property
    def drone_count(self) -> int:
        return len(self._drones)

    @property
    def online_count(self) -> int:
        return sum(1 for s in self._drones.values() if s.online)
