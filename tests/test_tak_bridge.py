"""Tests for TakBridge."""

import pytest

from sentinel.common.event_bus import EventBus
from sentinel.common.time import utc_now_iso
from sentinel.common.types import (
    GeoObservation,
    GeoPoint,
    MissionPlan,
    VehicleState,
)
from sentinel.config.schema import TakConfig
from sentinel.tak_bridge.bridge import TakBridge


def _vehicle():
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80.0),
        heading_deg=90.0,
        mode="SIMULATED",
        armed=False,
        battery_pct=90.0,
    )


def _obs():
    return GeoObservation(
        observation_id="obs_001",
        mission_id="m1",
        track_id="trk_000001",
        class_name="person",
        timestamp_utc=utc_now_iso(),
        estimated_location=GeoPoint(lat=38.001, lon=-8.001, alt_m=0.0),
        accuracy_estimate_m=35.0,
        confidence=0.85,
    )


def _plan():
    return MissionPlan(
        mission_id="m1",
        waypoints=[
            GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
            GeoPoint(lat=38.01, lon=-8.0, alt_m=80),
        ],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
    )


def test_bridge_sends_vehicle_state_creates_jsonl(tmp_path):
    cfg = TakConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bridge = TakBridge(config=cfg, mission_id="m1", run_dir=run_dir)
    result = bridge.send_vehicle_state(_vehicle())
    bridge.close()
    assert result is True
    assert (run_dir / "tak_messages.jsonl").exists()


def test_bridge_send_geo_observation_writes_category(tmp_path):
    cfg = TakConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bridge = TakBridge(config=cfg, mission_id="m1", run_dir=run_dir)
    bridge.send_geo_observation(_obs())
    bridge.close()
    import json

    lines = (run_dir / "tak_messages.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["category"] == "observation"


def test_bridge_send_mission_plan_returns_count(tmp_path):
    cfg = TakConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bridge = TakBridge(config=cfg, mission_id="m1", run_dir=run_dir)
    sent = bridge.send_mission_plan(_plan())
    bridge.close()
    assert sent == 2


def test_bridge_close_creates_metrics(tmp_path):
    cfg = TakConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bridge = TakBridge(config=cfg, mission_id="m1", run_dir=run_dir)
    bridge.send_vehicle_state(_vehicle())
    bridge.close()
    assert (run_dir / "tak_metrics.json").exists()


def test_bridge_publishes_tak_message_sent():
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)
    cfg = TakConfig()
    bridge = TakBridge(config=cfg, mission_id="m1", event_bus=bus)
    bridge.send_vehicle_state(_vehicle())
    bridge.close()
    types = [e.event_type for e in events]
    assert "tak_message_sent" in types


def test_bridge_publishes_tak_bridge_completed():
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)
    cfg = TakConfig()
    bridge = TakBridge(config=cfg, mission_id="m1", event_bus=bus)
    bridge.send_vehicle_state(_vehicle())
    bridge.close()
    types = [e.event_type for e in events]
    assert "tak_bridge_completed" in types


def test_invalid_mode_fails():
    with pytest.raises(ValueError, match="Unsupported TAK mode"):
        TakConfig(mode="tcp")
