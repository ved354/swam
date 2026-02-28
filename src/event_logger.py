"""
VayuSwarm — Persistent Event Logger (SQLite)

Stores detection events, telemetry snapshots, commands, and safety vetoes
in a local SQLite database for post-mission analysis and audit trail.
"""

from __future__ import annotations

import json
import sqlite3
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Default DB path (relative to project root)
DEFAULT_DB_PATH = "data/vayuswarm_events.db"


class EventLogger:
    """
    Thread-safe SQLite event logger.

    Records:
      - detection_events: fused detections with timestamp, drone_id, class, confidence
      - telemetry_log: periodic telemetry snapshots
      - commands: ground→drone commands
      - safety_vetoes: safety layer interventions
      - audit_log: free-form system events (startup, shutdown, errors)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """Open (or create) the database and ensure schema exists."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("event_logger.opened", db_path=self._db_path)

    def _create_tables(self) -> None:
        assert self._conn
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                drone_id TEXT NOT NULL,
                detection_class TEXT,
                confidence REAL,
                threat_level TEXT,
                lat REAL,
                lon REAL,
                alt REAL,
                payload TEXT
            );

            CREATE TABLE IF NOT EXISTS telemetry_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                drone_id TEXT NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                heading REAL,
                speed REAL,
                battery_pct REAL,
                state TEXT,
                mode TEXT,
                armed INTEGER
            );

            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                source_id TEXT,
                target_drone_id TEXT,
                command_type TEXT,
                priority TEXT,
                lat REAL,
                lon REAL,
                alt REAL,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS safety_vetoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                drone_id TEXT NOT NULL,
                reason TEXT,
                original_action TEXT,
                override_action TEXT,
                details TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                level TEXT NOT NULL,
                component TEXT,
                message TEXT,
                data TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_det_ts ON detection_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_det_drone ON detection_events(drone_id);
            CREATE INDEX IF NOT EXISTS idx_telem_ts ON telemetry_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_telem_drone ON telemetry_log(drone_id);
            CREATE INDEX IF NOT EXISTS idx_cmd_ts ON commands(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
        """)

    # ─── Insert methods ─────────────────────────────────────────────────────

    def log_detection(
        self,
        drone_id: str,
        detection_class: str,
        confidence: float,
        threat_level: str = "NONE",
        lat: float = 0.0,
        lon: float = 0.0,
        alt: float = 0.0,
        payload: Optional[dict] = None,
    ) -> None:
        """Log a detection event."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT INTO detection_events
                   (timestamp, drone_id, detection_class, confidence,
                    threat_level, lat, lon, alt, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), drone_id, detection_class, confidence,
                    threat_level, lat, lon, alt,
                    json.dumps(payload) if payload else None,
                ),
            )
            self._conn.commit()

    def log_telemetry(
        self,
        drone_id: str,
        lat: float,
        lon: float,
        alt: float,
        heading: float = 0.0,
        speed: float = 0.0,
        battery_pct: float = 0.0,
        state: str = "",
        mode: str = "",
        armed: bool = False,
    ) -> None:
        """Log a telemetry snapshot."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT INTO telemetry_log
                   (timestamp, drone_id, lat, lon, alt, heading, speed,
                    battery_pct, state, mode, armed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), drone_id, lat, lon, alt, heading, speed,
                    battery_pct, state, mode, int(armed),
                ),
            )
            self._conn.commit()

    def log_command(
        self,
        source_id: str,
        target_drone_id: str,
        command_type: str,
        priority: str = "NORMAL",
        lat: float = 0.0,
        lon: float = 0.0,
        alt: float = 0.0,
        message: str = "",
    ) -> None:
        """Log a ground command."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT INTO commands
                   (timestamp, source_id, target_drone_id, command_type,
                    priority, lat, lon, alt, message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), source_id, target_drone_id, command_type,
                    priority, lat, lon, alt, message,
                ),
            )
            self._conn.commit()

    def log_safety_veto(
        self,
        drone_id: str,
        reason: str,
        original_action: str = "",
        override_action: str = "",
        details: str = "",
    ) -> None:
        """Log a safety veto event."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT INTO safety_vetoes
                   (timestamp, drone_id, reason, original_action,
                    override_action, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (time.time(), drone_id, reason, original_action,
                 override_action, details),
            )
            self._conn.commit()

    def log_audit(
        self,
        level: str,
        component: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        """Log a free-form audit event."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT INTO audit_log
                   (timestamp, level, component, message, data)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), level, component, message,
                 json.dumps(data) if data else None),
            )
            self._conn.commit()

    # ─── Query methods ──────────────────────────────────────────────────────

    def get_detections(
        self,
        drone_id: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query detection events."""
        with self._lock:
            if not self._conn:
                return []
            query = "SELECT * FROM detection_events WHERE 1=1"
            params: list = []
            if drone_id:
                query += " AND drone_id = ?"
                params.append(drone_id)
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, params)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_telemetry_history(
        self,
        drone_id: str,
        since: Optional[float] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query telemetry history for a drone."""
        with self._lock:
            if not self._conn:
                return []
            query = "SELECT * FROM telemetry_log WHERE drone_id = ?"
            params: list = [drone_id]
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, params)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_audit_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        with self._lock:
            if not self._conn:
                return []
            cursor = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged data."""
        with self._lock:
            if not self._conn:
                return {}
            counts = {}
            for table in ("detection_events", "telemetry_log", "commands",
                          "safety_vetoes", "audit_log"):
                row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0] if row else 0
            return counts

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        logger.info("event_logger.closed")
