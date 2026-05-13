"""Tests for the mission recorder."""

import json
from pathlib import Path

from sentinel.common.events import make_event
from sentinel.recorder.recorder import MissionRecorder


def test_recorder_creates_run_dir(tmp_path: Path):
    rec = MissionRecorder(tmp_path, "test_mission")
    run_dir = rec.start_run()

    assert run_dir.exists()
    assert (run_dir / "metadata.json").exists()

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["mission_id"] == "test_mission"

    event = make_event("test_mission", "test", "test_event")
    rec.record_event(event)
    rec.close()

    lines = (run_dir / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["event_type"] == "test_event"


def test_close_idempotent(tmp_path: Path):
    rec = MissionRecorder(tmp_path, "m1")
    rec.start_run()
    rec.close()
    rec.close()
