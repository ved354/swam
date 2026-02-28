"""
VayuSwarm — Dashboard API Routes

REST API and WebSocket endpoints for the real-time dashboard.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

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
