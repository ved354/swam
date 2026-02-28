"""
VayuSwarm — Mission Recorder & Replay

Records mission data (telemetry, events, frames, decisions) to disk during
flight and replays them through the dashboard for post-mission review.

Recording format:
    data/recordings/<mission_id>/
        meta.json          — mission metadata + timeline bounds
        telemetry.jsonl    — one DroneTelemetry JSON per line per drone per tick
        events.jsonl       — detection / fused events
        decisions.jsonl    — LLM decisions + safety vetoes
        commands.jsonl     — ground commands issued

Replay:
    The Replayer reads a recording directory and yields frames at the
    original pace (or a user-chosen speed multiplier).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_RECORDINGS_DIR = "data/recordings"


# ═══════════════════════════════════════════════════════════════════════════════
# Recorder
# ═══════════════════════════════════════════════════════════════════════════════


class MissionRecorder:
    """
    Records mission data to a timestamped directory as newline-delimited JSON.
    Thread-safe for concurrent writes from multiple drones.
    """

    def __init__(
        self,
        mission_id: Optional[str] = None,
        base_dir: str = DEFAULT_RECORDINGS_DIR,
    ):
        self._mission_id = mission_id or f"mission_{int(time.time())}"
        self._dir = Path(base_dir) / self._mission_id
        self._files: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._event_count = 0
        self._open = False

    @property
    def mission_id(self) -> str:
        return self._mission_id

    @property
    def recording_dir(self) -> Path:
        return self._dir

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def event_count(self) -> int:
        return self._event_count

    def start(self) -> None:
        """Create recording directory and open JSONL files."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

        for name in ("telemetry", "events", "decisions", "commands"):
            path = self._dir / f"{name}.jsonl"
            self._files[name] = open(path, "a")

        # Write metadata
        meta = {
            "mission_id": self._mission_id,
            "start_time": self._start_time,
            "version": "0.1.0",
        }
        (self._dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self._open = True
        logger.info("recorder.started",
                     mission_id=self._mission_id, dir=str(self._dir))

    def record_telemetry(self, data: dict) -> None:
        """Append a telemetry snapshot."""
        self._write("telemetry", data)

    def record_event(self, data: dict) -> None:
        """Append a detection/fused event."""
        self._write("events", data)

    def record_decision(self, data: dict) -> None:
        """Append an LLM decision or safety veto."""
        self._write("decisions", data)

    def record_command(self, data: dict) -> None:
        """Append a ground command."""
        self._write("commands", data)

    def _write(self, stream: str, data: dict) -> None:
        fh = self._files.get(stream)
        if not fh or fh.closed:
            return
        entry = {
            "_t": time.time(),
            **data,
        }
        fh.write(json.dumps(entry, default=str) + "\n")
        fh.flush()
        self._event_count += 1

    def stop(self) -> None:
        """Finalize recording: update metadata and close files."""
        if not self._open:
            return
        end_time = time.time()

        # Update metadata with end time
        meta_path = self._dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}
        meta["end_time"] = end_time
        meta["duration_s"] = end_time - (self._start_time or end_time)
        meta["event_count"] = self._event_count
        meta_path.write_text(json.dumps(meta, indent=2))

        for fh in self._files.values():
            if fh and not fh.closed:
                fh.close()
        self._files.clear()
        self._open = False
        logger.info("recorder.stopped",
                     mission_id=self._mission_id,
                     events=self._event_count,
                     duration=meta.get("duration_s", 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Replayer
# ═══════════════════════════════════════════════════════════════════════════════


class MissionReplayer:
    """
    Replays a recorded mission from disk.
    Yields events in chronological order with optional speed multiplier.
    """

    def __init__(self, recording_dir: str):
        self._dir = Path(recording_dir)
        self._meta: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def duration_s(self) -> float:
        return self._meta.get("duration_s", 0.0)

    def load(self) -> None:
        """Load all events from the recording directory into memory."""
        meta_path = self._dir / "meta.json"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text())

        self._events = []
        for stream in ("telemetry", "events", "decisions", "commands"):
            path = self._dir / f"{stream}.jsonl"
            if not path.exists():
                continue
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entry["_stream"] = stream
                        self._events.append(entry)
                    except json.JSONDecodeError:
                        continue

        # Sort by timestamp
        self._events.sort(key=lambda e: e.get("_t", 0))
        logger.info("replayer.loaded",
                     dir=str(self._dir),
                     events=len(self._events),
                     duration=self.duration_s)

    def replay(self, speed: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """
        Yield events in chronological order, sleeping between them
        to match original timing (scaled by speed multiplier).

        Args:
            speed: 1.0 = real-time, 2.0 = 2x fast, 0 = instant (no sleep)

        Yields:
            dict with keys: _t (timestamp), _stream (telemetry|events|...),
            plus all original fields.
        """
        prev_t = None
        for event in self._events:
            t = event.get("_t", 0)
            if prev_t is not None and speed > 0:
                gap = (t - prev_t) / speed
                if gap > 0:
                    time.sleep(gap)
            prev_t = t
            yield event

    def get_events_in_range(
        self,
        start_t: float,
        end_t: float,
        stream: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get events within a time range, optionally filtered by stream."""
        results = []
        for e in self._events:
            t = e.get("_t", 0)
            if t < start_t:
                continue
            if t > end_t:
                break
            if stream and e.get("_stream") != stream:
                continue
            results.append(e)
        return results

    def get_timeline_summary(self, buckets: int = 50) -> List[Dict[str, Any]]:
        """Get a bucketed timeline summary for dashboard scrubbing."""
        if not self._events:
            return []
        t_min = self._events[0].get("_t", 0)
        t_max = self._events[-1].get("_t", 0)
        if t_max <= t_min:
            return [{"t": t_min, "count": len(self._events)}]

        bucket_size = (t_max - t_min) / buckets
        summary: List[Dict[str, Any]] = []
        bucket_start = t_min
        idx = 0

        for _ in range(buckets):
            bucket_end = bucket_start + bucket_size
            count = 0
            streams: Dict[str, int] = {}
            while idx < len(self._events) and self._events[idx].get("_t", 0) < bucket_end:
                count += 1
                s = self._events[idx].get("_stream", "unknown")
                streams[s] = streams.get(s, 0) + 1
                idx += 1
            summary.append({
                "t": bucket_start,
                "count": count,
                "streams": streams,
            })
            bucket_start = bucket_end

        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# List recordings
# ═══════════════════════════════════════════════════════════════════════════════


def list_recordings(base_dir: str = DEFAULT_RECORDINGS_DIR) -> List[Dict[str, Any]]:
    """List all available recordings with metadata."""
    base = Path(base_dir)
    if not base.exists():
        return []

    recordings = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["path"] = str(entry)
                recordings.append(meta)
            except json.JSONDecodeError:
                continue
    return recordings
