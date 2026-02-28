"""
VayuSwarm — Message Protocol Schemas

All inter-component communication uses strongly-typed Pydantic models.
These schemas define the contract between ground station, drones, vision
pipeline, safety layer, and PX4 interface.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enumerations ───────────────────────────────────────────────────────────────

class ThreatLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectionClass(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    VEHICLE_CAR = "vehicle_car"
    VEHICLE_TRUCK = "vehicle_truck"
    VEHICLE_MOTORCYCLE = "vehicle_motorcycle"
    WEAPON_RIFLE = "weapon_rifle"
    WEAPON_HANDGUN = "weapon_handgun"
    WEAPON_PISTOL = "weapon_pistol"
    WEAPON_KNIFE = "weapon_knife"
    DRONE = "drone"
    FIRE = "fire"
    SUSPICIOUS_PACKAGE = "suspicious_package"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


class UniformType(str, Enum):
    MILITARY = "military"
    CIVILIAN = "civilian"
    UNKNOWN = "unknown"


class BehaviorType(str, Enum):
    STATIONARY = "stationary"
    PATROL = "patrol"
    EVASIVE_MOVEMENT = "evasive_movement"
    APPROACHING = "approaching"
    RETREATING = "retreating"
    FORMATION = "formation"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"


class DroneState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    TAKEOFF = "TAKEOFF"
    PATROL = "PATROL"
    INVESTIGATE = "INVESTIGATE"
    TRACK = "TRACK"
    ENGAGE = "ENGAGE"
    RTL = "RTL"           # Return to Launch
    LAND = "LAND"
    EMERGENCY = "EMERGENCY"
    OFFLINE = "OFFLINE"


class CommandType(str, Enum):
    GOTO_WAYPOINT = "goto_waypoint"
    SET_PATROL_ZONE = "set_patrol_zone"
    INVESTIGATE_TARGET = "investigate_target"
    TRACK_TARGET = "track_target"
    RETURN_TO_LAUNCH = "return_to_launch"
    LAND = "land"
    HOLD_POSITION = "hold_position"
    CHANGE_ALTITUDE = "change_altitude"
    UPDATE_ROE = "update_roe"
    EMERGENCY_STOP = "emergency_stop"
    FORMATION_CHANGE = "formation_change"


class CommandPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectionSource(str, Enum):
    RGB = "rgb"
    THERMAL = "thermal"
    FUSED = "fused"


class SafetyVetoReason(str, Enum):
    NGZ_VIOLATION = "ngz_violation"              # No-go zone
    ALTITUDE_LIMIT = "altitude_limit"
    BATTERY_CRITICAL = "battery_critical"
    COLLISION_RISK = "collision_risk"
    COMMS_LOST = "comms_lost"
    ROE_VIOLATION = "roe_violation"
    LLM_ANOMALY = "llm_anomaly"
    GEOFENCE_BREACH = "geofence_breach"


# ─── Base Message ───────────────────────────────────────────────────────────────

class BaseMessage(BaseModel):
    """Base for all VayuSwarm messages."""
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    source_id: str = Field(..., description="ID of the sender (drone_id or 'ground')")


# ─── Geospatial ─────────────────────────────────────────────────────────────────

class GeoPoint(BaseModel):
    """A geographic coordinate with optional altitude."""
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    alt: float = Field(default=0.0, description="Altitude in meters MSL")


class BoundingBox(BaseModel):
    """2D bounding box in image coordinates (normalized 0-1)."""
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)


class GeoZone(BaseModel):
    """A geographic zone defined by a polygon of points."""
    zone_id: str
    name: str
    points: list[GeoPoint] = Field(..., min_length=3, description="Polygon vertices")
    is_no_go: bool = Field(default=False)
    max_altitude: Optional[float] = None
    min_altitude: Optional[float] = None


# ─── Telemetry ──────────────────────────────────────────────────────────────────

class DroneTelemetry(BaseMessage):
    """Real-time telemetry from a drone."""
    drone_id: str
    position: GeoPoint
    heading: float = Field(..., ge=0, lt=360, description="Heading in degrees")
    speed: float = Field(..., ge=0, description="Ground speed in m/s")
    vertical_speed: float = Field(default=0.0, description="Vertical speed in m/s")
    battery_pct: float = Field(..., ge=0, le=100)
    battery_voltage: float = Field(default=0.0)
    gps_fix: int = Field(default=0, description="GPS fix type (0=none, 3=3D)")
    satellites: int = Field(default=0)
    armed: bool = Field(default=False)
    state: DroneState = Field(default=DroneState.IDLE)
    mode: str = Field(default="GUIDED")
    signal_strength: float = Field(default=100.0, ge=0, le=100)


# ─── Detection Events ──────────────────────────────────────────────────────────

class DetectionEvent(BaseModel):
    """A single detection from one sensor source."""
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    source: DetectionSource
    detection_class: DetectionClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    geo_position: Optional[GeoPoint] = None
    metadata: dict = Field(default_factory=dict)


class ThermalDetection(BaseModel):
    """Detection from thermal camera specifically."""
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    blob_class: DetectionClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    temperature_c: float = Field(default=0.0, description="Estimated temperature in Celsius")
    blob_area: float = Field(default=0.0, description="Relative blob area (0-1)")
    geo_position: Optional[GeoPoint] = None
    group_count: int = Field(default=1, description="Estimated number of entities in blob")
    group_spacing_m: float = Field(default=0.0, description="Avg spacing between entities (meters)")


class FusedEvent(BaseModel):
    """Merged multi-sensor detection — the primary input to the Transformer/LLM."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    # Classification
    detection_class: DetectionClass
    class_confidence: float = Field(..., ge=0.0, le=1.0)
    # Armed status
    armed: bool = Field(default=False)
    armed_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    weapon_class: Optional[DetectionClass] = None
    # Uniform
    uniform: UniformType = Field(default=UniformType.UNKNOWN)
    uniform_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Location
    geo_position: Optional[GeoPoint] = None
    bbox: Optional[BoundingBox] = None
    # Thermal
    temperature_c: Optional[float] = None
    group_count: int = Field(default=1)
    # Behavior (filled in by BehaviorAnalyzer)
    behavior: BehaviorType = Field(default=BehaviorType.UNKNOWN)
    behavior_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    movement_speed_ms: float = Field(default=0.0)
    movement_heading: float = Field(default=0.0)
    # Threat
    threat_level: ThreatLevel = Field(default=ThreatLevel.NONE)
    in_ngz: bool = Field(default=False)
    # Sources
    sources: list[DetectionSource] = Field(default_factory=list)

    def to_llm_text(self) -> str:
        """Convert to structured text for LLM consumption."""
        parts = [
            f"{self.detection_class.value} detected",
            f"{self.class_confidence:.0%} confidence",
            f"sources: {'+'.join(s.value for s in self.sources)}",
            f"{self.threat_level.value} threat",
        ]
        if self.armed:
            parts.append(f"ARMED ({self.weapon_class.value if self.weapon_class else 'unknown'}, "
                         f"{self.armed_confidence:.0%})")
        if self.uniform != UniformType.UNKNOWN:
            parts.append(f"uniform: {self.uniform.value} ({self.uniform_confidence:.0%})")
        if self.behavior != BehaviorType.UNKNOWN:
            parts.append(f"behavior: {self.behavior.value}")
        if self.in_ngz:
            parts.append("IN NO-GO ZONE")
        if self.geo_position:
            parts.append(f"at ({self.geo_position.lat:.6f}, {self.geo_position.lon:.6f})")
        return ", ".join(parts)


# ─── Drone Reports ─────────────────────────────────────────────────────────────

class DroneReport(BaseMessage):
    """Aggregated report from a drone to the ground station."""
    drone_id: str
    telemetry: DroneTelemetry
    fused_events: list[FusedEvent] = Field(default_factory=list)
    local_llm_decision: Optional[str] = None
    safety_vetoes: list[str] = Field(default_factory=list)
    current_action: str = Field(default="idle")
    notes: str = Field(default="")


# ─── Ground Commands ────────────────────────────────────────────────────────────

class GroundCommand(BaseMessage):
    """Strategic command from ground station to a drone."""
    target_drone_id: str
    command_type: CommandType
    priority: CommandPriority = Field(default=CommandPriority.NORMAL)
    # Optional parameters depending on command
    waypoint: Optional[GeoPoint] = None
    patrol_zone: Optional[GeoZone] = None
    target_event_id: Optional[str] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    formation_index: Optional[int] = None
    roe_update: Optional[dict] = None
    message: str = Field(default="", description="Natural language directive from ground LLM")


# ─── Swarm Messages ────────────────────────────────────────────────────────────

class SwarmMessage(BaseMessage):
    """Drone-to-drone coordination message."""
    sender_drone_id: str
    # Share position
    position: Optional[GeoPoint] = None
    heading: Optional[float] = None
    speed: Optional[float] = None
    # Target handoff
    handoff_event: Optional[FusedEvent] = None
    # Formation
    formation_position: Optional[int] = None
    # Free text
    message: str = Field(default="")


# ─── Safety Veto ────────────────────────────────────────────────────────────────

class SafetyVeto(BaseModel):
    """Safety layer override — blocks an unsafe action."""
    veto_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    drone_id: str
    reason: SafetyVetoReason
    original_action: str = Field(..., description="The action that was blocked")
    override_action: str = Field(..., description="The safe action taken instead")
    severity: ThreatLevel = Field(default=ThreatLevel.HIGH)
    details: str = Field(default="")


# ─── Mission ────────────────────────────────────────────────────────────────────

class MissionObjective(BaseModel):
    """A single mission objective."""
    objective_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    priority: CommandPriority = Field(default=CommandPriority.NORMAL)
    target_zone: Optional[GeoZone] = None
    completed: bool = Field(default=False)


class MissionDefinition(BaseModel):
    """Full mission definition."""
    mission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: float = Field(default_factory=time.time)
    objectives: list[MissionObjective] = Field(default_factory=list)
    patrol_zones: list[GeoZone] = Field(default_factory=list)
    no_go_zones: list[GeoZone] = Field(default_factory=list)
    roe: dict = Field(default_factory=lambda: {
        "engagement_allowed": False,
        "warning_first": True,
        "min_confidence_to_alert": 0.7,
        "min_confidence_to_track": 0.5,
        "auto_rtl_battery_pct": 20,
        "max_altitude_m": 120,
        "min_altitude_m": 10,
    })
    drone_ids: list[str] = Field(default_factory=list)
    active: bool = Field(default=True)


# ─── LLM Decision ──────────────────────────────────────────────────────────────

class LLMDecision(BaseModel):
    """Structured decision output from local or ground LLM."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    source: str = Field(..., description="'local_llm' or 'ground_llm'")
    action: str = Field(..., description="The decided action")
    reasoning: str = Field(default="", description="LLM's reasoning chain")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    target_event_id: Optional[str] = None
    suggested_waypoint: Optional[GeoPoint] = None
    suggested_altitude: Optional[float] = None
    suggested_speed: Optional[float] = None
    raw_response: str = Field(default="")
