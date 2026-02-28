"""
VayuSwarm — Swarm Coordinator

Provides intelligent multi-drone coordination:

1. ZONE ASSIGNMENT     — Divides the patrol area into N sectors,
                         assigns one sector per drone. No overlap.

2. DYNAMIC REBALANCING — When a drone goes offline or RTLs, its
                         zone is redistributed among remaining drones.

3. TARGET HANDOFF      — When tracking drone goes low battery, it
                         nominates the nearest healthy drone to take over.

4. COLLISION AVOIDANCE — Each drone's intended waypoint is checked
                         against all other drone positions; if too close,
                         the waypoint is deflected to maintain separation.

5. SWARM CONSENSUS     — When multiple drones detect the same target,
                         one is elected as the tracker (lowest distance);
                         others resume patrol.

Usage (from GroundStation):
    coordinator = SwarmCoordinator(config)
    coordinator.update_fleet(fleet_manager)           # call every cycle
    commands = coordinator.tick(fleet_manager, mission_planner)
    for cmd in commands:
        await station.send_command(cmd)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from proto.messages import (
    CommandPriority,
    CommandType,
    DroneState,
    FusedEvent,
    GeoPoint,
    GeoZone,
    GroundCommand,
    ThreatLevel,
)

logger = structlog.get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

COLLISION_SAFE_DIST_M   = 30.0    # metres — minimum horizontal separation
HANDOFF_BATTERY_PCT     = 25.0    # % — trigger handoff below this
HANDOFF_BATTERY_HYSTER  = 5.0     # % — hysteresis to avoid flip-flop
ZONE_REBALANCE_COOLDOWN = 30.0    # seconds between rebalance events
CONSENSUS_SAME_TARGET_M = 20.0    # metres — two reports = same target if closer
PATROL_ALT_M            = 60.0    # default patrol altitude metres AGL

# ─── Haversine helper ─────────────────────────────────────────────────────────

def _haversine(a: GeoPoint, b: GeoPoint) -> float:
    """Return horizontal distance in metres between two GeoPoints."""
    R = 6_371_000.0
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _midpoint(a: GeoPoint, b: GeoPoint) -> GeoPoint:
    return GeoPoint(lat=(a.lat + b.lat) / 2, lon=(a.lon + b.lon) / 2, alt=a.alt)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PatrolSector:
    """A rectangular sector of the overall patrol polygon."""
    sector_id: str
    assigned_drone: Optional[str]
    waypoints: list[GeoPoint]          # lawnmower path for this sector
    center: GeoPoint


@dataclass
class TrackedTarget:
    """A confirmed target being monitored by the swarm."""
    target_id: str
    position: GeoPoint
    threat: ThreatLevel
    tracking_drone: Optional[str]      # which drone owns tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float  = field(default_factory=time.time)
    handoff_pending: bool = False


# ─── Zone Divider ─────────────────────────────────────────────────────────────

class ZoneDivider:
    """
    Splits a patrol polygon into N equal-area sectors using a grid-based
    horizontal stripe + column approach.

    For N drones over a bounding box:
      - Compute lat/lon bounding box of the patrol zone
      - Divide into a grid of N tiles (as square as possible)
      - Generate a lawnmower waypoint path within each tile
    """

    @staticmethod
    def divide(zone_points: list[GeoPoint], n: int, alt: float = PATROL_ALT_M) -> list[PatrolSector]:
        if n <= 0 or not zone_points:
            return []

        lats = [p.lat for p in zone_points]
        lons = [p.lon for p in zone_points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        # Compute grid dimensions (as square as possible)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        dlat = (lat_max - lat_min) / rows
        dlon = (lon_max - lon_min) / cols

        sectors = []
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                s_lat = lat_min + row * dlat
                e_lat = s_lat + dlat
                s_lon = lon_min + col * dlon
                e_lon = s_lon + dlon

                center = GeoPoint(lat=(s_lat + e_lat) / 2, lon=(s_lon + e_lon) / 2, alt=alt)

                # Lawnmower: 3 parallel strips within the sector
                waypoints = ZoneDivider._lawnmower(s_lat, e_lat, s_lon, e_lon, alt, strips=3)

                sectors.append(PatrolSector(
                    sector_id=f"sector_{idx:02d}",
                    assigned_drone=None,
                    waypoints=waypoints,
                    center=center,
                ))
                idx += 1

        return sectors

    @staticmethod
    def _lawnmower(lat_s: float, lat_e: float, lon_s: float, lon_e: float,
                   alt: float, strips: int = 3) -> list[GeoPoint]:
        """Generate a lawnmower (boustrophedon) path within a rectangular sector."""
        step = (lon_e - lon_s) / (strips - 1) if strips > 1 else 0
        wps = []
        for i in range(strips):
            lon = lon_s + i * step
            if i % 2 == 0:
                wps.append(GeoPoint(lat=lat_s, lon=lon, alt=alt))
                wps.append(GeoPoint(lat=lat_e, lon=lon, alt=alt))
            else:
                wps.append(GeoPoint(lat=lat_e, lon=lon, alt=alt))
                wps.append(GeoPoint(lat=lat_s, lon=lon, alt=alt))
        return wps


# ─── Collision Avoidance ──────────────────────────────────────────────────────

class CollisionAvoider:
    """
    Checks a proposed waypoint against all other drone positions.
    If the proposed point is within COLLISION_SAFE_DIST_M of any other
    drone, deflects the waypoint outward until safe.
    """

    @staticmethod
    def check_and_deflect(
        drone_id: str,
        proposed: GeoPoint,
        all_positions: dict[str, GeoPoint],
        safe_dist_m: float = COLLISION_SAFE_DIST_M,
    ) -> GeoPoint:
        result = proposed
        for other_id, other_pos in all_positions.items():
            if other_id == drone_id:
                continue
            dist = _haversine(result, other_pos)
            if dist < safe_dist_m:
                # Deflect: push result away from other_pos
                delta_lat = result.lat - other_pos.lat
                delta_lon = result.lon - other_pos.lon
                # If positions are identical, push north by default
                if delta_lat == 0.0 and delta_lon == 0.0:
                    delta_lat = 1e-6
                norm = math.sqrt(delta_lat ** 2 + delta_lon ** 2) + 1e-9
                # How much to move in degrees (approx: 1 deg lat ≈ 111_000 m)
                push_m = safe_dist_m - dist + 5.0
                push_lat = (delta_lat / norm) * (push_m / 111_000)
                push_lon = (delta_lon / norm) * (push_m / (111_000 * math.cos(math.radians(result.lat)) + 1e-9))
                result = GeoPoint(lat=result.lat + push_lat, lon=result.lon + push_lon, alt=result.alt)
                logger.debug("collision.deflected", drone=drone_id, other=other_id, dist=round(dist, 1))
        return result


# ─── Swarm Consensus ──────────────────────────────────────────────────────────

class SwarmConsensus:
    """
    When multiple drones report detecting the same physical target:
    - Clusters reports within CONSENSUS_SAME_TARGET_M
    - Elects ONE tracker (closest drone)
    - Issues TRACK command to winner, RESUME_PATROL to others
    """

    @staticmethod
    def resolve(
        drone_reports: dict,     # drone_id → DroneReport
        all_positions: dict[str, GeoPoint],
        active_targets: dict[str, TrackedTarget],
    ) -> tuple[list[GroundCommand], dict[str, TrackedTarget]]:
        """
        Returns:
            commands        — list of GroundCommand to issue
            active_targets  — updated target registry
        """
        commands: list[GroundCommand] = []

        # Collect all high-threat detections across all drones
        candidates: list[tuple[str, GeoPoint, ThreatLevel]] = []   # (drone_id, position, threat)
        for drone_id, report in drone_reports.items():
            for event in getattr(report, "fused_events", []):
                if event.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                    if event.gps_position:
                        candidates.append((drone_id, event.gps_position, event.threat_level))

        if not candidates:
            return commands, active_targets

        # Cluster candidates into unique physical targets
        clusters: list[list[tuple]] = []
        for cand in candidates:
            placed = False
            for cluster in clusters:
                if _haversine(cand[1], cluster[0][1]) < CONSENSUS_SAME_TARGET_M:
                    cluster.append(cand)
                    placed = True
                    break
            if not placed:
                clusters.append([cand])

        now = time.time()

        for cluster in clusters:
            # Compute centroid position
            centroid_lat = sum(c[1].lat for c in cluster) / len(cluster)
            centroid_lon = sum(c[1].lon for c in cluster) / len(cluster)
            centroid     = GeoPoint(lat=centroid_lat, lon=centroid_lon, alt=PATROL_ALT_M)

            # Is this an already-tracked target?
            existing_id = None
            for tid, tgt in active_targets.items():
                if _haversine(centroid, tgt.position) < CONSENSUS_SAME_TARGET_M * 2:
                    existing_id = tid
                    break

            target_id = existing_id or f"tgt_{int(now)}_{len(active_targets)}"
            highest_threat = max(c[2] for c in cluster)

            # Elect tracker: drone closest to the centroid
            reporter_drones = {c[0] for c in cluster}
            best_drone = None
            best_dist  = float("inf")
            for drone_id in reporter_drones:
                pos = all_positions.get(drone_id)
                if pos:
                    d = _haversine(pos, centroid)
                    if d < best_dist:
                        best_dist = d
                        best_drone = drone_id

            if not best_drone:
                continue

            # Update target registry
            if existing_id and existing_id in active_targets:
                active_targets[existing_id].position    = centroid
                active_targets[existing_id].last_seen   = now
                active_targets[existing_id].tracking_drone = best_drone
            else:
                active_targets[target_id] = TrackedTarget(
                    target_id=target_id,
                    position=centroid,
                    threat=highest_threat,
                    tracking_drone=best_drone,
                )

            # Issue TRACK to elected drone
            commands.append(GroundCommand(
                source_id="ground",
                target_drone_id=best_drone,
                command_type=CommandType.TRACK_TARGET,
                priority=CommandPriority.HIGH,
                waypoint=centroid,
                parameters={"target_id": target_id, "reason": "consensus_elected"},
            ))
            logger.info("consensus.tracker_elected",
                         target=target_id,
                         tracker=best_drone,
                         reporters=len(reporter_drones))

            # Issue HOLD / resume patrol to other reporters
            for drone_id in reporter_drones:
                if drone_id != best_drone:
                    commands.append(GroundCommand(
                        source_id="ground",
                        target_drone_id=drone_id,
                        command_type=CommandType.HOLD_POSITION,
                        priority=CommandPriority.NORMAL,
                        parameters={"reason": "consensus_yielded", "target_id": target_id},
                    ))

        return commands, active_targets


# ─── SwarmCoordinator — main class ────────────────────────────────────────────

class SwarmCoordinator:
    """
    Central swarm intelligence. Call `tick()` every ground-station loop cycle.

    Integrates zone assignment, rebalancing, target handoff,
    collision avoidance, and swarm consensus into a single coherent system.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get("swarm", {})
        self._safe_dist_m          = cfg.get("safe_dist_m",      COLLISION_SAFE_DIST_M)
        self._handoff_battery      = cfg.get("handoff_battery",  HANDOFF_BATTERY_PCT)
        self._rebalance_cooldown   = cfg.get("rebalance_cooldown", ZONE_REBALANCE_COOLDOWN)

        self._sectors:         list[PatrolSector]         = []
        self._active_targets:  dict[str, TrackedTarget]   = {}
        self._handoff_pending: dict[str, str]             = {}  # drone_id → target_id
        self._last_rebalance:  float                      = 0.0
        self._zone_initialized: bool                      = False
        self._drone_wp_index:   dict[str, int]            = {}  # drone → next wp index

        self._avoider   = CollisionAvoider()
        self._consensus = SwarmConsensus()

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize_zones(self, patrol_zone: GeoZone, drone_ids: list[str]) -> None:
        """Divide patrol zone and assign sectors to drones."""
        n  = len(drone_ids)
        self._sectors = ZoneDivider.divide(patrol_zone.points, n)
        for i, sector in enumerate(self._sectors):
            sector.assigned_drone     = drone_ids[i] if i < n else None
            self._drone_wp_index[drone_ids[i]] = 0
        self._zone_initialized = True
        logger.info("swarm.zones_initialized", n_sectors=len(self._sectors), drones=drone_ids)

    def tick(self, fleet_manager, mission_planner=None) -> list[GroundCommand]:
        """
        Called every ground-station loop iteration (~1–2 s).
        Returns list of GroundCommands to dispatch.
        """
        commands: list[GroundCommand] = []

        all_positions = fleet_manager.get_all_positions()
        online_ids    = fleet_manager.get_online_drone_ids()
        all_reports   = fleet_manager.get_all_reports()

        if not online_ids:
            return commands

        # 1. Initialize zones on first tick if mission planner available
        if not self._zone_initialized and mission_planner:
            patrol_zones = getattr(mission_planner, "patrol_zones", [])
            if patrol_zones:
                self.initialize_zones(patrol_zones[0], online_ids)

        # 2. Zone rebalancing — detect offline drones and redistribute
        commands += self._rebalance_zones(online_ids, all_positions)

        # 3. Patrol waypoint advancement — send next lawnmower wp to each drone
        commands += self._advance_patrol_waypoints(online_ids, all_positions, all_reports)

        # 4. Target handoff — low-battery trackers hand off to nearest healthy drone
        commands += self._process_handoffs(fleet_manager, all_positions, all_reports)

        # 5. Swarm consensus — multi-drone target deconfliction
        new_cmds, self._active_targets = self._consensus.resolve(
            all_reports, all_positions, self._active_targets
        )
        commands += new_cmds

        # 6. Collision avoidance — deflect any GOTO waypoints that would cause conflict
        commands = self._apply_collision_avoidance(commands, all_positions)

        # 7. Expire old targets
        self._expire_targets()

        return commands

    def get_sector_assignment(self) -> dict[str, str]:
        """Returns {drone_id: sector_id} mapping."""
        return {s.assigned_drone: s.sector_id for s in self._sectors if s.assigned_drone}

    def get_active_targets(self) -> list[dict]:
        """Returns serializable list of tracked targets."""
        return [
            {
                "target_id":      t.target_id,
                "lat":            t.position.lat,
                "lon":            t.position.lon,
                "threat":         t.threat.value,
                "tracking_drone": t.tracking_drone,
                "age_s":          round(time.time() - t.first_seen, 1),
            }
            for t in self._active_targets.values()
        ]

    # ── Internal methods ──────────────────────────────────────────────────────

    def _rebalance_zones(self, online_ids: list[str], all_positions: dict[str, GeoPoint]) -> list[GroundCommand]:
        """Detect drones that went offline and reassign their sectors."""
        if not self._sectors:
            return []

        now = time.time()
        if now - self._last_rebalance < self._rebalance_cooldown:
            return []

        # Find sectors whose assigned drone is now offline
        orphaned = [s for s in self._sectors if s.assigned_drone and s.assigned_drone not in online_ids]
        if not orphaned:
            return []

        self._last_rebalance = now
        commands: list[GroundCommand] = []

        # Unassigned online drones or least-loaded online drone gets orphaned sectors
        for sector in orphaned:
            lost_drone = sector.assigned_drone
            # Find online drone with fewest assigned sectors
            load: dict[str, int] = {d: 0 for d in online_ids}
            for s in self._sectors:
                if s.assigned_drone in load:
                    load[s.assigned_drone] += 1
            new_drone = min(load, key=load.get)
            sector.assigned_drone = new_drone
            self._drone_wp_index[new_drone] = 0

            logger.info("swarm.zone_rebalanced",
                         lost=lost_drone, sector=sector.sector_id, new_drone=new_drone)

            # Command the new drone to the sector's first waypoint
            if sector.waypoints:
                commands.append(GroundCommand(
                    source_id="ground",
                    target_drone_id=new_drone,
                    command_type=CommandType.GOTO_WAYPOINT,
                    priority=CommandPriority.HIGH,
                    waypoint=sector.waypoints[0],
                    parameters={"reason": "zone_rebalance", "sector": sector.sector_id},
                ))
        return commands

    def _advance_patrol_waypoints(
        self,
        online_ids: list[str],
        all_positions: dict[str, GeoPoint],
        all_reports: dict,
    ) -> list[GroundCommand]:
        """
        For each drone in PATROL state and close to its current sector waypoint,
        issue the next waypoint in the lawnmower path.
        """
        commands: list[GroundCommand] = []

        for sector in self._sectors:
            drone_id = sector.assigned_drone
            if not drone_id or drone_id not in online_ids:
                continue

            report = all_reports.get(drone_id)
            if not report:
                continue

            # Only advance if drone is in PATROL state
            state = getattr(report.telemetry, "state", None)
            if state not in (DroneState.PATROL, DroneState.IDLE):
                continue

            pos   = all_positions.get(drone_id)
            wps   = sector.waypoints
            if not pos or not wps:
                continue

            idx = self._drone_wp_index.get(drone_id, 0)
            if idx >= len(wps):
                idx = 0  # loop the lawnmower path

            current_wp = wps[idx]
            dist       = _haversine(pos, current_wp)

            # If within 15m of current wp → advance to next
            if dist < 15.0:
                idx = (idx + 1) % len(wps)
                self._drone_wp_index[drone_id] = idx
                next_wp = wps[idx]
                commands.append(GroundCommand(
                    source_id="ground",
                    target_drone_id=drone_id,
                    command_type=CommandType.GOTO_WAYPOINT,
                    priority=CommandPriority.LOW,
                    waypoint=next_wp,
                    parameters={"reason": "lawnmower_advance", "sector": sector.sector_id},
                ))

        return commands

    def _process_handoffs(
        self,
        fleet_manager,
        all_positions: dict[str, GeoPoint],
        all_reports: dict,
    ) -> list[GroundCommand]:
        """
        Check all tracking drones for low battery → hand target to nearest healthy drone.
        """
        commands: list[GroundCommand] = []
        online_ids = fleet_manager.get_online_drone_ids()

        for target_id, target in list(self._active_targets.items()):
            tracker = target.tracking_drone
            if not tracker or target.handoff_pending:
                continue

            status = fleet_manager.get_drone_status(tracker)
            if not status:
                continue

            battery = status.battery
            if battery > self._handoff_battery + HANDOFF_BATTERY_HYSTER:
                continue  # Still healthy

            # Find nearest healthy replacement drone
            candidate = None
            best_dist = float("inf")
            for drone_id in online_ids:
                if drone_id == tracker:
                    continue
                rep_status = fleet_manager.get_drone_status(drone_id)
                if not rep_status or rep_status.battery < self._handoff_battery + 15:
                    continue  # Too low to take over
                rep_report = all_reports.get(drone_id)
                if rep_report and getattr(rep_report.telemetry, "state", None) == DroneState.TRACK:
                    continue  # Already tracking something
                pos = all_positions.get(drone_id)
                if pos:
                    d = _haversine(pos, target.position)
                    if d < best_dist:
                        best_dist = d
                        candidate = drone_id

            if not candidate:
                logger.warning("swarm.handoff_no_candidate",
                                target=target_id, tracker=tracker, battery=battery)
                continue

            # Mark handoff pending to avoid re-triggering
            target.handoff_pending  = True
            target.tracking_drone   = candidate

            # Tell new drone to track
            commands.append(GroundCommand(
                source_id="ground",
                target_drone_id=candidate,
                command_type=CommandType.TRACK_TARGET,
                priority=CommandPriority.HIGH,
                waypoint=target.position,
                parameters={"target_id": target_id, "reason": "handoff_received"},
            ))

            # Tell old drone to RTL
            commands.append(GroundCommand(
                source_id="ground",
                target_drone_id=tracker,
                command_type=CommandType.RETURN_TO_LAUNCH,
                priority=CommandPriority.HIGH,
                parameters={"reason": "handoff_low_battery", "target_id": target_id},
            ))

            logger.info("swarm.handoff_executed",
                         target=target_id,
                         from_drone=tracker,
                         to_drone=candidate,
                         battery=round(battery, 1))

        return commands

    def _apply_collision_avoidance(
        self,
        commands: list[GroundCommand],
        all_positions: dict[str, GeoPoint],
    ) -> list[GroundCommand]:
        """
        Scan outgoing GOTO_WAYPOINT commands and deflect any that would
        bring a drone within COLLISION_SAFE_DIST_M of another drone.
        """
        result    = []
        sim_positions = dict(all_positions)  # copy so we can update with intended positions

        for cmd in commands:
            if cmd.command_type == CommandType.GOTO_WAYPOINT and cmd.waypoint:
                safe_wp = self._avoider.check_and_deflect(
                    cmd.target_drone_id,
                    cmd.waypoint,
                    sim_positions,
                    self._safe_dist_m,
                )
                # Update simulated position so subsequent commands in same tick are aware
                sim_positions[cmd.target_drone_id] = safe_wp

                if safe_wp.lat != cmd.waypoint.lat or safe_wp.lon != cmd.waypoint.lon:
                    # Rebuild command with deflected waypoint
                    params = dict(cmd.parameters or {})
                    params["collision_deflected"] = True
                    result.append(GroundCommand(
                        source_id="ground",
                        target_drone_id=cmd.target_drone_id,
                        command_type=cmd.command_type,
                        priority=cmd.priority,
                        waypoint=safe_wp,
                        parameters=params,
                    ))
                else:
                    result.append(cmd)
            else:
                result.append(cmd)

        return result

    def _expire_targets(self, max_age_s: float = 120.0) -> None:
        """Remove targets not seen in the last max_age_s seconds."""
        now  = time.time()
        dead = [tid for tid, t in self._active_targets.items()
                if now - t.last_seen > max_age_s]
        for tid in dead:
            logger.info("swarm.target_expired", target=tid)
            del self._active_targets[tid]
