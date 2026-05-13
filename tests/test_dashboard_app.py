"""Tests for dashboard FastAPI application."""

import json

import pytest
from fastapi.testclient import TestClient

from sentinel.dashboard.app import create_app


@pytest.fixture
def app_with_runs(tmp_path):
    """Create a test app with a sample run directory."""
    run_dir = tmp_path / "20260511T001000Z_demo_001"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"mission_id": "demo_001"}), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "test", "severity": "info", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "geo_observations.jsonl").write_text(
        json.dumps({
            "observation_id": "obs_1",
            "timestamp_utc": "2026-01-01T00:00:02Z",
            "class_name": "person",
            "confidence": 0.9,
            "estimated_location": {"lat": 38.0, "lon": -8.0},
            "metadata": {"frame_id": 0},
        }) + "\n",
        encoding="utf-8",
    )
    (run_dir / "vehicle_states.jsonl").write_text(
        json.dumps({
            "frame_id": 0,
            "timestamp_utc": "2026-01-01T00:00:02Z",
            "vehicle_id": "uav_001",
            "position": {"lat": 38.0, "lon": -8.0, "alt_m": 50.0},
            "heading_deg": 90.0,
            "groundspeed_mps": 12.0,
            "battery_pct": 88.0,
            "mode": "AUTO",
            "armed": True,
            "stale": False,
        }) + "\n",
        encoding="utf-8",
    )
    (run_dir / "realtime_metrics.json").write_text(json.dumps({"total_frames": 30}), encoding="utf-8")
    (run_dir / "report.md").write_text("# Mission Report\n\nAll good.", encoding="utf-8")
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_000000.jpg").write_bytes(b"jpegdata")
    (run_dir / "latest.jpg").write_bytes(b"jpegdata")
    return create_app(base_dir=tmp_path)


@pytest.fixture
def client(app_with_runs):
    return TestClient(app_with_runs)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_runs(client):
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["run_id"] == "20260511T001000Z_demo_001"
    assert data[0]["mission_id"] == "demo_001"


def test_get_latest_run(client):
    resp = client.get("/api/runs/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "20260511T001000Z_demo_001"


def test_get_run_by_id(client):
    resp = client.get("/api/runs/20260511T001000Z_demo_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "20260511T001000Z_demo_001"
    assert data["artifacts"]["observations_count"] == 1


def test_get_timeline(client):
    resp = client.get("/api/runs/20260511T001000Z_demo_001/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert "timeline" in data
    assert len(data["timeline"]) >= 1


def test_get_map_layers(client):
    resp = client.get("/api/runs/20260511T001000Z_demo_001/map")
    assert resp.status_code == 200
    data = resp.json()
    assert "waypoints" in data
    assert "observations" in data
    assert len(data["observations"]) == 1


def test_get_report(client):
    resp = client.get("/api/runs/20260511T001000Z_demo_001/report")
    assert resp.status_code == 200
    assert "Mission Report" in resp.text


def test_get_missing_run(client):
    resp = client.get("/api/runs/nonexistent")
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ISR Operations Console" in resp.text


def test_get_replay(client):
    resp = client.get("/api/runs/20260511T001000Z_demo_001/replay")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["frames"]) == 1


def test_get_frame_image(client):
    resp = client.get("/api/runs/20260511T001000Z_demo_001/frames/frame_000000.jpg")
    assert resp.status_code == 200
