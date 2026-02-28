"""
VayuSwarm — Dashboard Server

FastAPI + WebSocket server for the real-time mission control dashboard.
Serves static files and provides REST API + WebSocket for live data.
Supports optional API-key authentication and Prometheus metrics endpoint.
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
from fastapi.responses import FileResponse, PlainTextResponse

from src.dashboard.routes import router, set_ground_station, broadcast_update
from src.dashboard.auth import APIKeyMiddleware

logger = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ─── Simple Prometheus-style metrics ────────────────────────────────────────
_metrics: dict = {
    "requests_total": 0,
    "ws_clients_current": 0,
    "ws_messages_sent": 0,
    "uptime_start": time.time(),
}


def get_metrics() -> dict:
    """Return current metrics snapshot."""
    return {**_metrics, "uptime_s": time.time() - _metrics["uptime_start"]}


def create_app(ground_station=None, api_key: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="VayuSwarm — Mission Control",
        description="Real-time swarm drone command center",
        version="0.1.0",
    )

    # Optional API-key auth
    app.add_middleware(APIKeyMiddleware, api_key=api_key)

    app.include_router(router)

    if ground_station:
        set_ground_station(ground_station)

    # Serve static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        _metrics["requests_total"] += 1
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "VayuSwarm Dashboard — static files not found"}

    @app.get("/healthz")
    async def healthz():
        """Health check endpoint (public, no auth)."""
        return {"status": "ok", "uptime_s": time.time() - _metrics["uptime_start"]}

    @app.get("/metrics")
    async def metrics():
        """Prometheus-compatible metrics endpoint."""
        m = get_metrics()
        lines = [
            "# HELP vayuswarm_uptime_seconds Server uptime in seconds",
            "# TYPE vayuswarm_uptime_seconds gauge",
            f'vayuswarm_uptime_seconds {m["uptime_s"]:.1f}',
            "# HELP vayuswarm_requests_total Total HTTP requests",
            "# TYPE vayuswarm_requests_total counter",
            f'vayuswarm_requests_total {m["requests_total"]}',
            "# HELP vayuswarm_ws_clients Current WebSocket clients",
            "# TYPE vayuswarm_ws_clients gauge",
            f'vayuswarm_ws_clients {m["ws_clients_current"]}',
            "# HELP vayuswarm_ws_messages_sent Total WS messages sent",
            "# TYPE vayuswarm_ws_messages_sent counter",
            f'vayuswarm_ws_messages_sent {m["ws_messages_sent"]}',
        ]

        # Add per-drone fleet metrics if ground station available
        if ground_station:
            fleet = ground_station.fleet.get_fleet_summary()
            lines.append("# HELP vayuswarm_drones_total Number of known drones")
            lines.append("# TYPE vayuswarm_drones_total gauge")
            lines.append(f"vayuswarm_drones_total {len(fleet)}")
            for d in fleet:
                did = d.get("drone_id", "unknown")
                bat = d.get("battery_pct", 0)
                lines.append(f'vayuswarm_drone_battery_pct{{drone_id="{did}"}} {bat}')

        return PlainTextResponse("\n".join(lines) + "\n",
                                  media_type="text/plain; version=0.0.4")

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
        api_key: Optional[str] = None,
    ):
        self._ground_station = ground_station
        self._host = host
        self._port = port
        self._update_interval = update_interval_ms / 1000.0
        self._app = create_app(ground_station, api_key=api_key)
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
