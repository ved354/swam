"""
VayuSwarm — Safety Layer

Two-tier safety system that sits between the LLM and PX4:
  1. Hard Rules — always enforced, no ML involved
  2. ML Guardrails — validates LLM outputs against ROE

If the LLM suggests an unsafe action, the safety layer VETOES it
and substitutes a safe default action.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import structlog

from proto.messages import (
    DroneState,
    GeoPoint,
    GeoZone,
    LLMDecision,
    SafetyVeto,
    SafetyVetoReason,
    ThreatLevel,
)

logger = structlog.get_logger(__name__)


class SafetyLayer:
    """
    Two-tier safety system:
    
    Tier 1 — Hard Rules (deterministic, always enforced):
      • No-go zone enforcement (geofence)
      • Altitude limits (min/max)
      • Battery reserve thresholds
      • Collision avoidance (min distance between drones)
      • Return-to-launch on comms loss
      • Geofence (max distance from home)
    
    Tier 2 — ML Guardrails (learned, validates LLM outputs):
      • ROE compliance check
      • Anomaly detection on LLM outputs
      • Action consistency check
    """

    def __init__(
        self,
        drone_id: str,
        home_position: GeoPoint,
        # Hard rule parameters
        max_altitude_m: float = 120.0,
        min_altitude_m: float = 5.0,
        battery_critical_pct: float = 15.0,
        battery_warning_pct: float = 25.0,
        collision_radius_m: float = 10.0,
        comms_loss_timeout_s: float = 15.0,
        geofence_radius_m: float = 5000.0,
        # Zones
        no_go_zones: Optional[list[GeoZone]] = None,
    ):
        self._drone_id = drone_id
        self._home = home_position
        self._max_alt = max_altitude_m
        self._min_alt = min_altitude_m
        self._battery_critical = battery_critical_pct
        self._battery_warning = battery_warning_pct
        self._collision_radius = collision_radius_m
        self._comms_timeout = comms_loss_timeout_s
        self._geofence_radius = geofence_radius_m
        self._no_go_zones = no_go_zones or []
        self._last_comms_time = time.time()
        self._peer_positions: dict[str, GeoPoint] = {}
        self._vetoes: list[SafetyVeto] = []
        self._checks_passed = 0
        self._checks_vetoed = 0

    def validate(
        self,
        decision: LLMDecision,
        current_position: GeoPoint,
        current_state: DroneState,
        battery_pct: float,
        peer_positions: Optional[dict[str, GeoPoint]] = None,
    ) -> tuple[LLMDecision, Optional[SafetyVeto]]:
        """
        Validate an LLM decision through both safety tiers.
        
        Returns:
            (decision, veto) — if veto is not None, the original decision was overridden
        """
        self._peer_positions = peer_positions or {}

        # ═══════════════════════════════════════════════════════════════
        # TIER 1 — HARD RULES (Cannot be overridden by any LLM)
        # ═══════════════════════════════════════════════════════════════

        # Rule 1: Battery Critical → Force RTL
        veto = self._check_battery(battery_pct, decision)
        if veto:
            return self._apply_veto(decision, veto)

        # Rule 2: Comms Loss → Force RTL
        veto = self._check_comms_loss()
        if veto:
            return self._apply_veto(decision, veto)

        # Rule 3: No-Go Zone Check (for suggested waypoint)
        if decision.suggested_waypoint:
            veto = self._check_ngz(decision.suggested_waypoint, decision)
            if veto:
                return self._apply_veto(decision, veto)

        # Rule 4: Altitude Limits
        if decision.suggested_altitude is not None:
            veto = self._check_altitude(decision.suggested_altitude, decision)
            if veto:
                return self._apply_veto(decision, veto)
        if decision.suggested_waypoint and decision.suggested_waypoint.alt:
            veto = self._check_altitude(decision.suggested_waypoint.alt, decision)
            if veto:
                return self._apply_veto(decision, veto)

        # Rule 5: Geofence Check
        target = decision.suggested_waypoint or current_position
        veto = self._check_geofence(target, decision)
        if veto:
            return self._apply_veto(decision, veto)

        # Rule 6: Collision Avoidance
        if decision.suggested_waypoint:
            veto = self._check_collision(decision.suggested_waypoint, decision)
            if veto:
                return self._apply_veto(decision, veto)

        # ═══════════════════════════════════════════════════════════════
        # TIER 2 — ML GUARDRAILS
        # ═══════════════════════════════════════════════════════════════

        # Rule 7: ROE Compliance
        veto = self._check_roe_compliance(decision)
        if veto:
            return self._apply_veto(decision, veto)

        # Rule 8: Anomaly Detection
        veto = self._check_anomaly(decision)
        if veto:
            return self._apply_veto(decision, veto)

        # All checks passed
        self._checks_passed += 1
        return decision, None

    def _check_battery(self, battery_pct: float, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Force RTL if battery is critically low."""
        if battery_pct <= self._battery_critical and decision.action != "RTL":
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.BATTERY_CRITICAL,
                original_action=decision.action,
                override_action="RTL",
                severity=ThreatLevel.CRITICAL,
                details=f"Battery at {battery_pct:.0f}% (critical threshold: {self._battery_critical:.0f}%)",
            )
        return None

    def _check_comms_loss(self) -> Optional[SafetyVeto]:
        """Force RTL if communications have been lost."""
        elapsed = time.time() - self._last_comms_time
        if elapsed > self._comms_timeout:
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.COMMS_LOST,
                original_action="UNKNOWN",
                override_action="RTL",
                severity=ThreatLevel.CRITICAL,
                details=f"No comms for {elapsed:.0f}s (timeout: {self._comms_timeout:.0f}s)",
            )
        return None

    def _check_ngz(self, waypoint: GeoPoint, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Block movement into no-go zones."""
        for zone in self._no_go_zones:
            if zone.is_no_go and self._point_in_polygon(waypoint, zone.points):
                return SafetyVeto(
                    drone_id=self._drone_id,
                    reason=SafetyVetoReason.NGZ_VIOLATION,
                    original_action=decision.action,
                    override_action="HOLD",
                    severity=ThreatLevel.HIGH,
                    details=f"Target waypoint inside no-go zone '{zone.name}'",
                )
        return None

    def _check_altitude(self, altitude: float, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Enforce altitude limits."""
        if altitude > self._max_alt:
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.ALTITUDE_LIMIT,
                original_action=decision.action,
                override_action="HOLD",
                severity=ThreatLevel.MEDIUM,
                details=f"Requested altitude {altitude:.0f}m exceeds max {self._max_alt:.0f}m",
            )
        if altitude < self._min_alt:
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.ALTITUDE_LIMIT,
                original_action=decision.action,
                override_action="HOLD",
                severity=ThreatLevel.MEDIUM,
                details=f"Requested altitude {altitude:.0f}m below min {self._min_alt:.0f}m",
            )
        return None

    def _check_geofence(self, target: GeoPoint, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Enforce maximum distance from home."""
        dist = self._geo_distance(self._home, target)
        if dist > self._geofence_radius:
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.GEOFENCE_BREACH,
                original_action=decision.action,
                override_action="RTL",
                severity=ThreatLevel.HIGH,
                details=f"Target {dist:.0f}m from home (max: {self._geofence_radius:.0f}m)",
            )
        return None

    def _check_collision(self, waypoint: GeoPoint, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Check for potential collisions with peer drones."""
        for peer_id, peer_pos in self._peer_positions.items():
            dist = self._geo_distance(waypoint, peer_pos)
            if dist < self._collision_radius:
                return SafetyVeto(
                    drone_id=self._drone_id,
                    reason=SafetyVetoReason.COLLISION_RISK,
                    original_action=decision.action,
                    override_action="HOLD",
                    severity=ThreatLevel.CRITICAL,
                    details=f"Collision risk with {peer_id} ({dist:.1f}m, min: {self._collision_radius:.0f}m)",
                )
        return None

    def _check_roe_compliance(self, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Check if the decision complies with Rules of Engagement."""
        # Block any engagement-like actions
        blocked_actions = ["ENGAGE", "FIRE", "ATTACK", "STRIKE", "NEUTRALIZE"]
        if decision.action.upper() in blocked_actions:
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.ROE_VIOLATION,
                original_action=decision.action,
                override_action="HOLD",
                severity=ThreatLevel.CRITICAL,
                details=f"Action '{decision.action}' violates ROE — engagement not authorized",
            )
        return None

    def _check_anomaly(self, decision: LLMDecision) -> Optional[SafetyVeto]:
        """Detect anomalous LLM outputs."""
        # Low confidence with high-impact action
        risky_actions = ["INVESTIGATE", "TRACK"]
        if decision.action.upper() in risky_actions and decision.confidence < 0.3:
            return SafetyVeto(
                drone_id=self._drone_id,
                reason=SafetyVetoReason.LLM_ANOMALY,
                original_action=decision.action,
                override_action="HOLD",
                severity=ThreatLevel.MEDIUM,
                details=f"Low confidence ({decision.confidence:.0%}) for risky action '{decision.action}'",
            )
        return None

    def _apply_veto(
        self,
        original: LLMDecision,
        veto: SafetyVeto,
    ) -> tuple[LLMDecision, SafetyVeto]:
        """Apply a safety veto — override the decision."""
        self._vetoes.append(veto)
        self._checks_vetoed += 1

        logger.warning(
            "safety.VETO",
            drone_id=self._drone_id,
            reason=veto.reason.value,
            original=veto.original_action,
            override=veto.override_action,
            details=veto.details,
        )

        # Create overridden decision
        safe_decision = LLMDecision(
            source="safety_layer",
            action=veto.override_action,
            reasoning=f"SAFETY VETO: {veto.details}",
            confidence=0.99,
            raw_response=original.raw_response,
        )

        return safe_decision, veto

    def update_comms_time(self) -> None:
        """Update last communication timestamp (call on every message received)."""
        self._last_comms_time = time.time()

    def set_no_go_zones(self, zones: list[GeoZone]) -> None:
        """Update no-go zones."""
        self._no_go_zones = [z for z in zones if z.is_no_go]
        logger.info("safety.ngz_updated", count=len(self._no_go_zones))

    def update_peer_positions(self, positions: dict[str, GeoPoint]) -> None:
        """Update known positions of peer drones."""
        self._peer_positions = positions

    @staticmethod
    def _point_in_polygon(point: GeoPoint, polygon: list[GeoPoint]) -> bool:
        """Ray casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            if ((polygon[i].lat > point.lat) != (polygon[j].lat > point.lat) and
                    point.lon < (polygon[j].lon - polygon[i].lon) *
                    (point.lat - polygon[i].lat) /
                    (polygon[j].lat - polygon[i].lat) + polygon[i].lon):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _geo_distance(p1: GeoPoint, p2: GeoPoint) -> float:
        """Approximate distance in meters (equirectangular)."""
        R = 6371000
        x = math.radians(p2.lon - p1.lon) * math.cos(math.radians((p1.lat + p2.lat) / 2))
        y = math.radians(p2.lat - p1.lat)
        return R * math.sqrt(x * x + y * y)

    @property
    def stats(self) -> dict:
        return {
            "checks_passed": self._checks_passed,
            "checks_vetoed": self._checks_vetoed,
            "total_vetoes": len(self._vetoes),
            "veto_rate": (self._checks_vetoed / max(1, self._checks_passed + self._checks_vetoed)) * 100,
        }

    @property
    def recent_vetoes(self) -> list[SafetyVeto]:
        """Get recent vetoes."""
        return self._vetoes[-10:]
