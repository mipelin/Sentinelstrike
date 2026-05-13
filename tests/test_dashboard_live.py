"""Tests for LiveRunSession."""

import json
from pathlib import Path

from sentinel.dashboard.live import LiveRunSession


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def test_session_poll_returns_counts(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    session = LiveRunSession(run_id="run_001", run_dir=run_dir)
    payload = session.poll()
    assert payload["run_id"] == "run_001"
    assert "artifact_counts" in payload
    assert payload["artifact_counts"]["events"] == 0


def test_session_poll_reads_new_events(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    session = LiveRunSession(run_id="run_001", run_dir=run_dir)
    session.poll()  # initial empty poll

    _write_jsonl(run_dir / "events.jsonl", [
        {"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "test_event", "severity": "info"},
    ])
    payload = session.poll()
    assert payload["artifact_counts"]["events"] == 1


def test_session_poll_reads_geo_observations(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    session = LiveRunSession(run_id="run_001", run_dir=run_dir)
    _write_jsonl(run_dir / "geo_observations.jsonl", [
        {
            "observation_id": "obs_1",
            "timestamp_utc": "2026-01-01T00:00:01Z",
            "class_name": "person",
            "confidence": 0.9,
            "estimated_location": {"lat": 38.0, "lon": -8.0},
        },
    ])
    payload = session.poll()
    assert payload["artifact_counts"]["geo_observations"] == 1
    assert len(payload["map_layers"]["observations"]) == 1


def test_session_timeline_tail(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    session = LiveRunSession(run_id="run_001", run_dir=run_dir)
    events = [
        {"timestamp_utc": f"2026-01-01T00:00:0{i}Z", "event_type": f"ev_{i}", "severity": "info"}
        for i in range(5)
    ]
    _write_jsonl(run_dir / "events.jsonl", events)
    payload = session.poll()
    assert len(payload["timeline_tail"]) == 5


def test_session_reset(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    session = LiveRunSession(run_id="run_001", run_dir=run_dir)
    _write_jsonl(run_dir / "events.jsonl", [
        {"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "test", "severity": "info"},
    ])
    session.poll()
    assert session._tailers["events"].line_count == 1

    session.reset()
    assert session._tailers["events"].line_count == 0


def test_session_handles_missing_files(tmp_path):
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()
    session = LiveRunSession(run_id="run_empty", run_dir=run_dir)
    payload = session.poll()
    assert payload["run_id"] == "run_empty"
    assert all(v == 0 for v in payload["artifact_counts"].values())
