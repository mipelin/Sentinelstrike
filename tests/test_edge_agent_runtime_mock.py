"""Tests for EdgeAgentRuntime in mock mode."""

import json
from pathlib import Path

from sentinel.common.types import GeoPoint, MissionRequest
from sentinel.config.schema import AppConfig
from sentinel.edge_agent.runtime import EdgeAgentRuntime


def _request():
    return MissionRequest(
        mission_id="runtime_test",
        launch_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=0),
        area_of_interest=[
            GeoPoint(lat=38.000, lon=-8.000, alt_m=80),
            GeoPoint(lat=38.000, lon=-7.998, alt_m=80),
            GeoPoint(lat=38.002, lon=-7.998, alt_m=80),
        ],
    )


def _config(tmp_path):
    return AppConfig(
        recorder={"output_dir": str(tmp_path / "runs"), "enabled": True},
        perception={"backend": "mock", "max_frames": 15, "source": str(tmp_path / "vid.mp4"), "classes": ["person"]},
        tracker={"enabled": True},
        geolocalizer={"enabled": True, "camera": {"pitch_deg": -45.0}},
        tak={"enabled": True, "mode": "dry_run"},
    )


def _ensure_video(tmp_path):
    import cv2
    import numpy as np

    vid_path = tmp_path / "vid.mp4"
    if vid_path.exists():
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(vid_path), fourcc, 20.0, (320, 240))
    for _ in range(15):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()


def _run_mock(tmp_path) -> dict:
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    runtime = EdgeAgentRuntime(config=cfg, mission_request=_request(), mode="mock", max_frames=15)
    return runtime.run()


def test_runtime_creates_run_dir(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert run_dir.exists()


def test_runtime_creates_mission_plan(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "mission_plan.json").exists()


def test_runtime_creates_detections_jsonl(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "detections.jsonl").exists()


def test_runtime_creates_tracks_jsonl(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "tracks.jsonl").exists()


def test_runtime_creates_geo_observations_jsonl(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "geo_observations.jsonl").exists()


def test_runtime_creates_tak_messages_jsonl(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "tak_messages.jsonl").exists()


def test_runtime_creates_edge_agent_health(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "edge_agent_health.json").exists()


def test_runtime_creates_report(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "report.md").exists()


def test_events_contain_started(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    types = set()
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        evt = json.loads(line)
        types.add(evt["event_type"])
    assert "edge_agent_started" in types


def test_events_contain_completed(tmp_path):
    summary = _run_mock(tmp_path)
    run_dir = Path(summary["run_dir"])
    types = set()
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        evt = json.loads(line)
        types.add(evt["event_type"])
    assert "edge_agent_completed" in types


def test_summary_has_non_negative_counts(tmp_path):
    summary = _run_mock(tmp_path)
    assert summary["perception"]["detection_count"] >= 0
    assert summary["tracking"]["total_tracks"] >= 0
    assert summary["geolocalization"]["observation_count"] >= 0
    assert summary["tak"]["message_count"] >= 0
    assert summary["mavlink"]["command_count"] >= 0


def test_summary_has_edge_agent_section(tmp_path):
    summary = _run_mock(tmp_path)
    assert "edge_agent" in summary
    assert summary["edge_agent"]["mode"] == "mock"
    assert summary["edge_agent"]["lifecycle_final_state"] == "COMPLETED"
