"""Tests for UAV telemetry: vehicle_states.jsonl, map layers, mock movement."""

import json
from pathlib import Path

from sentinel.common.types import GeoPoint, VehicleState
from sentinel.dashboard.loader import load_run_artifacts
from sentinel.dashboard.map_layers import build_map_layers
from sentinel.realtime.mock_telemetry import MockMovementTelemetryProvider


def _make_vehicle_state(lat: float = 38.0, lon: float = -8.0, alt: float = 80.0,
                        heading: float = 90.0, speed: float = 10.0,
                        battery: float = 90.0, mode: str = "AUTO") -> dict:
    return {
        "timestamp_utc": "2026-05-11T12:00:00Z",
        "vehicle_id": "uav_001",
        "position": {"lat": lat, "lon": lon, "alt_m": alt},
        "heading_deg": heading,
        "groundspeed_mps": speed,
        "battery_pct": battery,
        "mode": mode,
        "armed": True,
        "stale": False,
    }


# --- MockMovementTelemetryProvider tests ---

def test_mock_movement_changes_position():
    provider = MockMovementTelemetryProvider(
        vehicle_id="uav_001",
        waypoints=[
            GeoPoint(lat=38.0, lon=-8.0),
            GeoPoint(lat=38.001, lon=-8.001),
        ],
    )
    s1 = provider.get_vehicle_state()
    s2 = provider.get_vehicle_state()
    assert s1.position is not None
    assert s2.position is not None
    # Position should change between calls
    assert s1.position.lat != s2.position.lat or s1.position.lon != s2.position.lon


def test_mock_movement_has_telemetry_fields():
    provider = MockMovementTelemetryProvider(vehicle_id="uav_001")
    state = provider.get_vehicle_state()
    assert state.vehicle_id == "uav_001"
    assert state.position is not None
    assert state.heading_deg is not None
    assert state.groundspeed_mps is not None
    assert state.battery_pct is not None
    assert state.mode == "AUTO"
    assert state.armed is True


def test_mock_movement_battery_decreases():
    provider = MockMovementTelemetryProvider(vehicle_id="uav_001")
    b1 = provider.get_vehicle_state().battery_pct
    for _ in range(50):
        provider.get_vehicle_state()
    b2 = provider.get_vehicle_state().battery_pct
    assert b2 < b1


def test_mock_movement_with_default_waypoints():
    provider = MockMovementTelemetryProvider(vehicle_id="uav_001", base_lat=38.0, base_lon=-8.0)
    states = [provider.get_vehicle_state() for _ in range(10)]
    # All positions should be near the base
    for s in states:
        assert abs(s.position.lat - 38.0) < 0.01
        assert abs(s.position.lon - (-8.0)) < 0.01


# --- Map layers with vehicle_states ---

def test_map_layers_extracts_uav_track():
    vs1 = _make_vehicle_state(lat=38.0, lon=-8.0)
    vs2 = _make_vehicle_state(lat=38.001, lon=-8.001)
    artifacts = {"vehicle_states": [vs1, vs2]}
    layers = build_map_layers(artifacts)

    assert len(layers.uav_positions) == 2
    assert len(layers.uav_track) == 2
    assert layers.uav_track[0].lat == 38.0
    assert layers.uav_track[1].lat == 38.001


def test_map_layers_latest_has_telemetry():
    vs = _make_vehicle_state(alt=100.0, heading=180.0, speed=15.0, battery=75.0, mode="GUIDED")
    artifacts = {"vehicle_states": [vs]}
    layers = build_map_layers(artifacts)

    pos = layers.uav_positions[0]
    assert pos.alt_m == 100.0
    assert pos.heading == 180.0
    assert pos.speed_mps == 15.0
    assert pos.battery_pct == 75.0
    assert pos.mode == "GUIDED"
    assert pos.armed is True
    assert pos.stale is False


def test_map_layers_stale_flag():
    vs = _make_vehicle_state()
    vs["stale"] = True
    artifacts = {"vehicle_states": [vs]}
    layers = build_map_layers(artifacts)

    assert layers.uav_positions[0].stale is True


def test_map_layers_empty_vehicle_states():
    artifacts = {"vehicle_states": []}
    layers = build_map_layers(artifacts)
    assert layers.uav_positions == []
    assert layers.uav_track == []


# --- vehicle_states.jsonl creation in run dir ---

def test_realtime_loop_creates_vehicle_states_jsonl(tmp_path: Path):
    from sentinel.realtime.loop import _vehicle_state_to_record

    vs = VehicleState(
        vehicle_id="uav_001",
        timestamp_utc="2026-05-11T12:00:00Z",
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80.0),
        heading_deg=90.0,
        groundspeed_mps=10.0,
        mode="AUTO",
        armed=True,
        battery_pct=90.0,
    )
    record = _vehicle_state_to_record(vs)
    assert record["vehicle_id"] == "uav_001"
    assert record["position"]["lat"] == 38.0
    assert record["heading_deg"] == 90.0
    assert record["battery_pct"] == 90.0
    assert record["stale"] is False


def test_realtime_loop_creates_stale_record():
    from sentinel.realtime.loop import _vehicle_state_to_record

    vs = VehicleState(
        vehicle_id="uav_001",
        timestamp_utc="2026-05-11T12:00:00Z",
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80.0),
        mode="UNKNOWN",
    )
    record = _vehicle_state_to_record(vs, stale=True)
    assert record["stale"] is True


# --- Loader integration ---

def test_loader_loads_vehicle_states(tmp_path: Path):
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"mission_id": "test"}))

    states = [
        _make_vehicle_state(lat=38.0, lon=-8.0),
        _make_vehicle_state(lat=38.001, lon=-8.001),
    ]
    (run_dir / "vehicle_states.jsonl").write_text(
        "\n".join(json.dumps(s) for s in states),
    )

    artifacts = load_run_artifacts(run_dir)
    assert len(artifacts["vehicle_states"]) == 2
    assert artifacts["vehicle_states"][0]["position"]["lat"] == 38.0


# --- Live session includes UAV data ---

def test_live_session_includes_uav_data(tmp_path: Path):
    from sentinel.dashboard.live import LiveRunSession

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Write some vehicle states
    states = [
        _make_vehicle_state(lat=38.0, lon=-8.0),
        _make_vehicle_state(lat=38.001, lon=-8.001),
    ]
    (run_dir / "vehicle_states.jsonl").write_text(
        "\n".join(json.dumps(s) for s in states) + "\n",
    )

    session = LiveRunSession(run_id="test_run", run_dir=run_dir)
    payload = session.poll()

    assert "latest_uav_state" in payload
    assert payload["latest_uav_state"] is not None
    assert payload["latest_uav_state"]["lat"] == 38.001
    assert "uav_track_tail" in payload
    assert len(payload["uav_track_tail"]) == 2
