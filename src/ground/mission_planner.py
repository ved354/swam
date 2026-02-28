"""
VayuSwarm — Mission Planner

Manages missions: patrol zones, no-go zones, objectives, ROE, and history.
"""

from __future__ import annotations

import time
from typing import Optional

import structlog

from proto.messages import (
    CommandPriority,
    GeoPoint,
    GeoZone,
    MissionDefinition,
    MissionObjective,
)

logger = structlog.get_logger(__name__)


class MissionPlanner:
    """
    Mission management system.
    
    Features:
    - Define and modify missions (zones, objectives, ROE)
    - Track mission progress
    - Maintain mission history
    """

    def __init__(self):
        self._active_mission: Optional[MissionDefinition] = None
        self._mission_history: list[MissionDefinition] = []

    def create_mission(
        self,
        name: str,
        drone_ids: list[str],
        patrol_zones: Optional[list[GeoZone]] = None,
        no_go_zones: Optional[list[GeoZone]] = None,
        objectives: Optional[list[dict]] = None,
        roe: Optional[dict] = None,
    ) -> MissionDefinition:
        """Create a new mission definition."""
        obj_list = []
        if objectives:
            for obj in objectives:
                obj_list.append(MissionObjective(
                    description=obj.get("description", ""),
                    priority=CommandPriority(obj.get("priority", "NORMAL")),
                ))

        mission = MissionDefinition(
            name=name,
            drone_ids=drone_ids,
            patrol_zones=patrol_zones or [],
            no_go_zones=no_go_zones or [],
            objectives=obj_list,
            roe=roe or MissionDefinition.model_fields["roe"].default_factory(),
        )

        self._active_mission = mission
        logger.info("mission.created",
                     name=name,
                     drones=len(drone_ids),
                     zones=len(patrol_zones or []),
                     objectives=len(obj_list))

        return mission

    def create_default_mission(self, drone_ids: list[str]) -> MissionDefinition:
        """Create a default surveillance mission around Hyderabad."""
        patrol_zone = GeoZone(
            zone_id="patrol_1",
            name="Primary Patrol Zone",
            points=[
                GeoPoint(lat=17.390, lon=78.480, alt=50),
                GeoPoint(lat=17.390, lon=78.495, alt=50),
                GeoPoint(lat=17.380, lon=78.495, alt=50),
                GeoPoint(lat=17.380, lon=78.480, alt=50),
            ],
        )

        no_go = GeoZone(
            zone_id="ngz_1",
            name="Restricted Area",
            points=[
                GeoPoint(lat=17.386, lon=78.488),
                GeoPoint(lat=17.386, lon=78.490),
                GeoPoint(lat=17.384, lon=78.490),
                GeoPoint(lat=17.384, lon=78.488),
            ],
            is_no_go=True,
            max_altitude=0,
        )

        return self.create_mission(
            name="Default Surveillance",
            drone_ids=drone_ids,
            patrol_zones=[patrol_zone],
            no_go_zones=[no_go],
            objectives=[
                {"description": "Monitor primary patrol zone", "priority": "HIGH"},
                {"description": "Detect and classify all persons in area", "priority": "NORMAL"},
                {"description": "Alert on HIGH+ threats", "priority": "CRITICAL"},
            ],
        )

    def complete_objective(self, objective_id: str) -> bool:
        """Mark an objective as completed."""
        if not self._active_mission:
            return False
        for obj in self._active_mission.objectives:
            if obj.objective_id == objective_id:
                obj.completed = True
                logger.info("mission.objective_completed", objective=obj.description)
                return True
        return False

    def complete_mission(self) -> None:
        """Complete the active mission and archive it."""
        if self._active_mission:
            self._active_mission.active = False
            self._mission_history.append(self._active_mission)
            logger.info("mission.completed", name=self._active_mission.name)
            self._active_mission = None

    def modify_roe(self, updates: dict) -> dict:
        """Modify rules of engagement for the active mission."""
        if self._active_mission:
            self._active_mission.roe.update(updates)
            logger.info("mission.roe_modified", updates=updates)
            return self._active_mission.roe
        return {}

    def add_no_go_zone(self, zone: GeoZone) -> None:
        """Add a no-go zone to the active mission."""
        if self._active_mission:
            zone.is_no_go = True
            self._active_mission.no_go_zones.append(zone)
            logger.info("mission.ngz_added", name=zone.name)

    @property
    def active_mission(self) -> Optional[MissionDefinition]:
        return self._active_mission

    @property
    def no_go_zones(self) -> list[GeoZone]:
        if self._active_mission:
            return self._active_mission.no_go_zones
        return []

    @property
    def patrol_zones(self) -> list[GeoZone]:
        if self._active_mission:
            return self._active_mission.patrol_zones
        return []

    @property
    def progress(self) -> dict:
        """Get mission progress."""
        if not self._active_mission:
            return {"active": False}

        total = len(self._active_mission.objectives)
        completed = sum(1 for o in self._active_mission.objectives if o.completed)
        return {
            "active": True,
            "name": self._active_mission.name,
            "objectives_total": total,
            "objectives_completed": completed,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
            "drones": len(self._active_mission.drone_ids),
        }
