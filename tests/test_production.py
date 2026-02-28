"""
VayuSwarm — Production Hardening Tests

Tests for:
  - EventLogger (SQLite persistence)
  - API key auth middleware
  - /healthz and /metrics endpoints
"""

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

from src.event_logger import EventLogger
from src.dashboard.server import create_app
from src.dashboard.auth import APIKeyMiddleware, check_ws_api_key


# ═══════════════════════════════════════════════════════════════════════════════
# EventLogger
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventLogger:
    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        self.logger = EventLogger(db_path=self._tmpfile.name)
        self.logger.open()

    def teardown_method(self):
        self.logger.close()
        os.unlink(self._tmpfile.name)

    def test_open_creates_tables(self):
        summary = self.logger.get_summary()
        for table in ("detection_events", "telemetry_log", "commands",
                       "safety_vetoes", "audit_log"):
            assert table in summary
            assert summary[table] == 0

    def test_log_detection_and_query(self):
        self.logger.log_detection(
            drone_id="drone-0",
            detection_class="person",
            confidence=0.92,
            threat_level="MEDIUM",
            lat=17.385, lon=78.487, alt=50.0,
        )
        rows = self.logger.get_detections(drone_id="drone-0")
        assert len(rows) == 1
        assert rows[0]["detection_class"] == "person"
        assert rows[0]["confidence"] == pytest.approx(0.92, abs=0.01)

    def test_log_telemetry_and_query(self):
        self.logger.log_telemetry(
            drone_id="drone-1",
            lat=17.0, lon=78.0, alt=50.0,
            heading=90.0, speed=5.0, battery_pct=80.0,
            state="PATROL", mode="OFFBOARD", armed=True,
        )
        rows = self.logger.get_telemetry_history("drone-1")
        assert len(rows) == 1
        assert rows[0]["state"] == "PATROL"
        assert rows[0]["armed"] == 1

    def test_log_command(self):
        self.logger.log_command(
            source_id="ground",
            target_drone_id="drone-0",
            command_type="goto_waypoint",
            priority="HIGH",
            lat=17.39, lon=78.49, alt=60.0,
            message="Investigate sector B",
        )
        summary = self.logger.get_summary()
        assert summary["commands"] == 1

    def test_log_safety_veto(self):
        self.logger.log_safety_veto(
            drone_id="drone-0",
            reason="battery_critical",
            original_action="GOTO",
            override_action="RTL",
        )
        summary = self.logger.get_summary()
        assert summary["safety_vetoes"] == 1

    def test_log_audit(self):
        self.logger.log_audit(
            level="INFO",
            component="agent",
            message="Drone started",
            data={"drone_id": "drone-0"},
        )
        rows = self.logger.get_audit_log()
        assert len(rows) == 1
        assert rows[0]["component"] == "agent"

    def test_query_with_since_filter(self):
        before = time.time()
        self.logger.log_detection(
            drone_id="d0", detection_class="person",
            confidence=0.9, lat=0, lon=0, alt=0,
        )
        rows = self.logger.get_detections(since=before - 1)
        assert len(rows) == 1
        rows_future = self.logger.get_detections(since=before + 100)
        assert len(rows_future) == 0

    def test_closed_logger_returns_empty(self):
        self.logger.close()
        assert self.logger.get_summary() == {}
        assert self.logger.get_detections() == []


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Auth, Healthz, Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardAuth:
    def test_no_auth_when_key_empty(self):
        app = create_app(api_key="")
        client = TestClient(app)
        resp = client.get("/api/fleet")
        # Without ground station → returns error json, but not 401
        assert resp.status_code == 200

    def test_auth_rejects_missing_key(self):
        app = create_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/api/fleet")
        assert resp.status_code == 401

    def test_auth_accepts_header_key(self):
        app = create_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/api/fleet", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200

    def test_auth_accepts_query_key(self):
        app = create_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/api/fleet?api_key=secret123")
        assert resp.status_code == 200

    def test_auth_rejects_wrong_key(self):
        app = create_app(api_key="secret123")
        client = TestClient(app)
        resp = client.get("/api/fleet", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


class TestHealthAndMetrics:
    def test_healthz_endpoint(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_s" in data

    def test_metrics_endpoint_prometheus_format(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "vayuswarm_uptime_seconds" in text
        assert "vayuswarm_requests_total" in text
        assert "vayuswarm_ws_clients" in text

    def test_metrics_content_type(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]


class TestWSAuthHelper:
    def test_check_ws_no_key_passes(self):
        assert check_ws_api_key(None, api_key=None) is True
        assert check_ws_api_key(None, api_key="") is True
