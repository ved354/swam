"""
VayuSwarm — Dashboard API Routes

REST API and WebSocket endpoints for the real-time dashboard.
Includes replay endpoints for post-mission review.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

from proto.messages import (
    CommandPriority,
    CommandType,
    GeoPoint,
    GroundCommand,
)
from src.recorder import MissionReplayer, list_recordings

logger = structlog.get_logger(__name__)

router = APIRouter()

# Will be set by server.py
_ground_station = None
_ws_clients: list[WebSocket] = []


def set_ground_station(station) -> None:
    """Set the ground station reference for API access."""
    global _ground_station
    _ground_station = station


@router.get("/api/fleet")
async def get_fleet():
    """Get fleet status summary."""
    if not _ground_station:
        return {"error": "Ground station not initialized"}
    return {
        "drones": _ground_station.fleet.get_fleet_summary(),
        "health": _ground_station.fleet.check_health(),
    }


@router.get("/api/events")
async def get_events():
    """Get recent detection events."""
    if not _ground_station:
        return {"error": "Ground station not initialized"}
    return {"events": _ground_station.event_log}


@router.get("/api/commands")
async def get_commands():
    """Get recent command log."""
    if not _ground_station:
        return {"error": "Ground station not initialized"}
    return {"commands": _ground_station.command_log}


@router.get("/api/mission")
async def get_mission():
    """Get current mission status."""
    if not _ground_station:
        return {"error": "Ground station not initialized"}
    return {
        "progress": _ground_station.mission_planner.progress,
        "mission": _ground_station.mission_planner.active_mission.model_dump()
        if _ground_station.mission_planner.active_mission else None,
    }


@router.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    if not _ground_station:
        return {"error": "Ground station not initialized"}
    return _ground_station.stats


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("dashboard.ws_client_connected", total=len(_ws_clients))

    try:
        while True:
            # Keep connection alive, handle incoming commands
            data = await websocket.receive_text()
            logger.debug("dashboard.ws_received", data=data[:100])

            # Process command messages from the frontend
            try:
                msg = json.loads(data)
                if msg.get("type") == "command" and _ground_station:
                    cmd = _build_command(msg)
                    if cmd:
                        await _ground_station.send_command(cmd)
                        await websocket.send_json({"type": "command_ack", "status": "ok"})
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        logger.info("dashboard.ws_client_disconnected", total=len(_ws_clients))


async def broadcast_update(data: dict) -> None:
    """Broadcast update to all connected WebSocket clients."""
    disconnected = []
    for client in _ws_clients:
        try:
            await client.send_json(data)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        _ws_clients.remove(client)


# ─── Command API ────────────────────────────────────────────────────────────────

_CMD_TYPE_MAP = {
    "goto_waypoint": CommandType.GOTO_WAYPOINT,
    "investigate": CommandType.INVESTIGATE_TARGET,
    "track": CommandType.TRACK_TARGET,
    "rtl": CommandType.RETURN_TO_LAUNCH,
    "land": CommandType.LAND,
    "hold": CommandType.HOLD_POSITION,
    "emergency_stop": CommandType.EMERGENCY_STOP,
    "change_altitude": CommandType.CHANGE_ALTITUDE,
}

_PRIORITY_MAP = {
    "low": CommandPriority.LOW,
    "normal": CommandPriority.NORMAL,
    "high": CommandPriority.HIGH,
    "critical": CommandPriority.CRITICAL,
}


def _build_command(payload: dict) -> Optional[GroundCommand]:
    """Build a GroundCommand from a dashboard payload dict."""
    drone_id = payload.get("drone_id", "")
    if not drone_id:
        return None

    cmd_str = payload.get("command", "hold").lower()
    cmd_type = _CMD_TYPE_MAP.get(cmd_str, CommandType.HOLD_POSITION)

    waypoint = None
    if payload.get("lat") and payload.get("lon"):
        waypoint = GeoPoint(
            lat=float(payload["lat"]),
            lon=float(payload["lon"]),
            alt=float(payload.get("alt", 50)),
        )

    priority = _PRIORITY_MAP.get(
        payload.get("priority", "normal").lower(), CommandPriority.NORMAL
    )

    return GroundCommand(
        source_id="dashboard",
        target_drone_id=drone_id,
        command_type=cmd_type,
        priority=priority,
        waypoint=waypoint,
        altitude=float(payload["alt"]) if payload.get("alt") else None,
        message=payload.get("message", "Manual command from dashboard"),
    )


@router.post("/api/command")
async def post_command(payload: dict):
    """Send a manual command from the dashboard."""
    if not _ground_station:
        return {"error": "Ground station not initialized"}

    try:
        cmd = _build_command(payload)
        if not cmd:
            return {"error": "Missing drone_id"}
        await _ground_station.send_command(cmd)
        return {"status": "ok", "command": cmd.command_type.value, "drone_id": cmd.target_drone_id}
    except Exception as e:
        logger.error("dashboard.command_error", error=str(e))
        return {"error": str(e)}


# ─── Replay API ─────────────────────────────────────────────────────────────────

@router.get("/api/recordings")
async def get_recordings():
    """List available mission recordings."""
    return {"recordings": list_recordings()}


@router.get("/api/recordings/{mission_id}")
async def get_recording_detail(mission_id: str):
    """Get details of a specific recording."""
    recordings = list_recordings()
    for rec in recordings:
        if rec.get("mission_id") == mission_id:
            replayer = MissionReplayer(rec["path"])
            replayer.load()
            return {
                "meta": replayer.meta,
                "total_events": replayer.total_events,
                "timeline": replayer.get_timeline_summary(),
            }
    return {"error": "Recording not found"}


@router.get("/api/recordings/{mission_id}/events")
async def get_recording_events(
    mission_id: str,
    start: float = 0,
    end: float = 0,
    stream: Optional[str] = None,
    limit: int = 200,
):
    """Get events from a recording within a time range."""
    recordings = list_recordings()
    for rec in recordings:
        if rec.get("mission_id") == mission_id:
            replayer = MissionReplayer(rec["path"])
            replayer.load()
            if end <= 0:
                end = float("inf")
            events = replayer.get_events_in_range(start, end, stream=stream)
            return {"events": events[:limit], "total": len(events)}
    return {"error": "Recording not found"}
