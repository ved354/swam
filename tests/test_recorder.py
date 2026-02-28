"""
VayuSwarm — Recorder & Replay Tests

Tests for MissionRecorder, MissionReplayer, and list_recordings.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.recorder import MissionRecorder, MissionReplayer, list_recordings


class TestMissionRecorder:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_start_creates_directory_and_meta(self):
        rec = MissionRecorder(mission_id="test_001", base_dir=self._tmpdir)
        rec.start()
        assert rec.is_open
        assert (rec.recording_dir / "meta.json").exists()
        meta = json.loads((rec.recording_dir / "meta.json").read_text())
        assert meta["mission_id"] == "test_001"
        rec.stop()

    def test_record_telemetry(self):
        rec = MissionRecorder(mission_id="tel_001", base_dir=self._tmpdir)
        rec.start()
        rec.record_telemetry({"drone_id": "d0", "lat": 17.0, "battery": 90})
        rec.record_telemetry({"drone_id": "d1", "lat": 18.0, "battery": 80})
        assert rec.event_count == 2
        rec.stop()

        # Verify file contents
        path = rec.recording_dir / "telemetry.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["drone_id"] == "d0"
        assert "_t" in entry

    def test_record_events_decisions_commands(self):
        rec = MissionRecorder(mission_id="mix_001", base_dir=self._tmpdir)
        rec.start()
        rec.record_event({"class": "person", "confidence": 0.9})
        rec.record_decision({"action": "INVESTIGATE", "drone": "d0"})
        rec.record_command({"type": "goto", "target": "d0"})
        assert rec.event_count == 3
        rec.stop()

        for name in ("events", "decisions", "commands"):
            path = rec.recording_dir / f"{name}.jsonl"
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1

    def test_stop_updates_metadata(self):
        rec = MissionRecorder(mission_id="stop_001", base_dir=self._tmpdir)
        rec.start()
        rec.record_telemetry({"x": 1})
        time.sleep(0.05)
        rec.stop()
        assert not rec.is_open

        meta = json.loads((rec.recording_dir / "meta.json").read_text())
        assert "end_time" in meta
        assert "duration_s" in meta
        assert meta["event_count"] == 1

    def test_stop_without_start_is_noop(self):
        rec = MissionRecorder(mission_id="noop", base_dir=self._tmpdir)
        rec.stop()  # Should not raise

    def test_write_after_stop_is_noop(self):
        rec = MissionRecorder(mission_id="closed", base_dir=self._tmpdir)
        rec.start()
        rec.stop()
        rec.record_telemetry({"x": 1})  # Should not raise or write
        assert rec.event_count == 0


class TestMissionReplayer:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def _create_recording(self, mission_id="replay_001", n_events=10):
        rec = MissionRecorder(mission_id=mission_id, base_dir=self._tmpdir)
        rec.start()
        for i in range(n_events):
            rec.record_telemetry({"drone_id": "d0", "i": i, "lat": 17 + i * 0.001})
            if i % 3 == 0:
                rec.record_event({"class": "person", "i": i})
        rec.stop()
        return str(rec.recording_dir)

    def test_load_recovers_all_events(self):
        path = self._create_recording(n_events=10)
        rp = MissionReplayer(path)
        rp.load()
        assert rp.total_events > 0
        assert rp.meta["mission_id"] == "replay_001"

    def test_events_sorted_by_timestamp(self):
        path = self._create_recording(n_events=5)
        rp = MissionReplayer(path)
        rp.load()
        timestamps = [e["_t"] for e in rp.get_events_in_range(0, float("inf"))]
        assert timestamps == sorted(timestamps)

    def test_replay_generator_yields_all(self):
        path = self._create_recording(n_events=5)
        rp = MissionReplayer(path)
        rp.load()
        events = list(rp.replay(speed=0))  # instant replay
        assert len(events) == rp.total_events

    def test_replay_preserves_stream_tag(self):
        path = self._create_recording(n_events=6)
        rp = MissionReplayer(path)
        rp.load()
        events = list(rp.replay(speed=0))
        streams = {e["_stream"] for e in events}
        assert "telemetry" in streams
        assert "events" in streams

    def test_get_events_in_range_filters(self):
        path = self._create_recording(n_events=20)
        rp = MissionReplayer(path)
        rp.load()
        all_events = rp.get_events_in_range(0, float("inf"))
        assert len(all_events) == rp.total_events

        # Filter by stream
        telem_only = rp.get_events_in_range(0, float("inf"), stream="telemetry")
        assert all(e["_stream"] == "telemetry" for e in telem_only)

    def test_timeline_summary(self):
        path = self._create_recording(n_events=20)
        rp = MissionReplayer(path)
        rp.load()
        summary = rp.get_timeline_summary(buckets=5)
        assert len(summary) == 5
        total = sum(b["count"] for b in summary)
        assert total == rp.total_events

    def test_empty_recording(self):
        rec = MissionRecorder(mission_id="empty", base_dir=self._tmpdir)
        rec.start()
        rec.stop()
        rp = MissionReplayer(str(rec.recording_dir))
        rp.load()
        assert rp.total_events == 0
        assert rp.get_timeline_summary() == []


class TestListRecordings:
    def test_list_empty_dir(self):
        tmpdir = tempfile.mkdtemp()
        assert list_recordings(base_dir=tmpdir) == []

    def test_list_finds_recordings(self):
        tmpdir = tempfile.mkdtemp()
        for mid in ("m1", "m2"):
            rec = MissionRecorder(mission_id=mid, base_dir=tmpdir)
            rec.start()
            rec.record_telemetry({"x": 1})
            rec.stop()
        recs = list_recordings(base_dir=tmpdir)
        assert len(recs) == 2
        ids = {r["mission_id"] for r in recs}
        assert ids == {"m1", "m2"}

    def test_list_nonexistent_dir(self):
        assert list_recordings(base_dir="/tmp/nonexistent_vayuswarm_xyz") == []
