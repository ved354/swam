"""
VayuSwarm — Dashboard Server

FastAPI + WebSocket server for the real-time mission control dashboard.
Serves static files and provides REST API + WebSocket for live data.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.dashboard.routes import router, set_ground_station, broadcast_update

logger = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(ground_station=None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="VayuSwarm — Mission Control",
        description="Real-time swarm drone command center",
        version="0.1.0",
    )

    app.include_router(router)

    if ground_station:
        set_ground_station(ground_station)

    # Serve static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "VayuSwarm Dashboard — static files not found"}

    return app


class DashboardServer:
    """
    Dashboard server with periodic WebSocket updates.
    """

    def __init__(
        self,
        ground_station=None,
        host: str = "0.0.0.0",
        port: int = 8080,
        update_interval_ms: int = 250,
    ):
        self._ground_station = ground_station
        self._host = host
        self._port = port
        self._update_interval = update_interval_ms / 1000.0
        self._app = create_app(ground_station)
        self._running = False

    async def start(self) -> None:
        """Start the dashboard server."""
        self._running = True

        # Start WebSocket update loop
        asyncio.create_task(self._update_loop())

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        logger.info("dashboard.starting", host=self._host, port=self._port)
        await server.serve()

    async def _update_loop(self) -> None:
        """Periodically broadcast fleet status via WebSocket."""
        while self._running:
            try:
                if self._ground_station:
                    data = {
                        "type": "update",
                        "timestamp": time.time(),
                        "fleet": self._ground_station.fleet.get_fleet_summary(),
                        "events": self._ground_station.event_log[-10:],
                        "commands": self._ground_station.command_log[-5:],
                        "mission": self._ground_station.mission_planner.progress,
                        "stats": self._ground_station.stats,
                    }
                    await broadcast_update(data)
            except Exception as e:
                logger.error("dashboard.update_error", error=str(e))

            await asyncio.sleep(self._update_interval)

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False
