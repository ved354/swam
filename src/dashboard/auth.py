"""
VayuSwarm — Dashboard Authentication Middleware

Simple API-key-based authentication for the dashboard API.
- Reads allowed keys from VAYUSWARM_API_KEY env var (or config)
- Static files and the index page are public
- All /api/* and /ws endpoints require a valid key

Usage:
    export VAYUSWARM_API_KEY="my-secret-key-123"
    python scripts/launch_dashboard.py

Or in config:
    dashboard:
      api_key: "my-secret-key-123"

Clients must send the key as:
    Header:  X-API-Key: my-secret-key-123
    Query:   /api/fleet?api_key=my-secret-key-123
    WS:      /ws?api_key=my-secret-key-123
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import structlog

logger = structlog.get_logger(__name__)


# Public paths that don't require auth
_PUBLIC_PREFIXES = ("/static", "/favicon.ico")
_PUBLIC_EXACT = ("/", "/index.html")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces API-key auth on /api/* endpoints.
    If no api_key is configured (None or empty), auth is disabled.
    """

    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        # Use explicit key, else env var, else disabled
        self._api_key = api_key or os.environ.get("VAYUSWARM_API_KEY", "")
        if self._api_key:
            logger.info("auth.api_key_enabled")
        else:
            logger.info("auth.api_key_disabled",
                        hint="Set VAYUSWARM_API_KEY to enable")

    async def dispatch(self, request: Request, call_next):
        # No key configured → pass through (dev mode)
        if not self._api_key:
            return await call_next(request)

        path = request.url.path

        # Public paths
        if path in _PUBLIC_EXACT or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        # Check auth for API routes
        if path.startswith("/api"):
            key = (
                request.headers.get("X-API-Key")
                or request.query_params.get("api_key")
            )
            if key != self._api_key:
                logger.warning("auth.rejected", path=path,
                               remote=request.client.host if request.client else "?")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )

        return await call_next(request)


def check_ws_api_key(websocket: WebSocket, api_key: Optional[str]) -> bool:
    """
    Check WebSocket connection for a valid API key.
    Call this in the WS endpoint before accepting.

    Returns True if auth passes (or auth is disabled).
    """
    if not api_key:
        return True
    key = websocket.query_params.get("api_key") or ""
    return key == api_key
