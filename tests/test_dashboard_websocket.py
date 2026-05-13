"""Tests for dashboard WebSocket endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from sentinel.dashboard.app import create_app


@pytest.fixture
def app_with_live(tmp_path):
    run_dir = tmp_path / "20260511T001000Z_demo_001"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"mission_id": "demo_001"}), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "test", "severity": "info", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    return create_app(base_dir=tmp_path)


@pytest.fixture
def client(app_with_live):
    return TestClient(app_with_live)


def test_ws_run_live_returns_payload(client):
    with client.websocket_connect("/ws/runs/20260511T001000Z_demo_001/live") as ws:
        data = ws.receive_json()
        assert data["run_id"] == "20260511T001000Z_demo_001"
        assert "artifact_counts" in data
        assert "timeline_tail" in data
        assert "map_layers" in data
        assert data["artifact_counts"]["events"] == 1


def test_ws_run_live_missing_run(client):
    with client.websocket_connect("/ws/runs/nonexistent/live") as ws:
        data = ws.receive_json()
        assert "error" in data


def test_ws_live_latest_returns_payload(client):
    with client.websocket_connect("/ws/live/latest") as ws:
        data = ws.receive_json()
        assert data["run_id"] == "20260511T001000Z_demo_001"
        assert "artifact_counts" in data


def test_ws_live_latest_no_runs(tmp_path):
    app = create_app(base_dir=tmp_path)
    c = TestClient(app)
    with c.websocket_connect("/ws/live/latest") as ws:
        data = ws.receive_json()
        assert data["status"] == "no_runs"


def test_ws_run_live_tracks_new_data(tmp_path):
    run_dir = tmp_path / "20260511T002000Z_demo_002"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"mission_id": "demo_002"}), encoding="utf-8")

    app = create_app(base_dir=tmp_path)
    c = TestClient(app)

    with c.websocket_connect("/ws/runs/20260511T002000Z_demo_002/live") as ws:
        # First poll — empty
        data1 = ws.receive_json()
        assert data1["artifact_counts"]["events"] == 0

        # Write new events
        (run_dir / "events.jsonl").write_text(
            json.dumps({"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "new_event", "severity": "info", "payload": {}}) + "\n",
            encoding="utf-8",
        )

        # Second poll should see it
        data2 = ws.receive_json()
        assert data2["artifact_counts"]["events"] == 1
