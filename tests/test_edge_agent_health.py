"""Tests for edge agent health snapshot."""

import json

from sentinel.common.event_bus import EventBus
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionRequest, VehicleState
from sentinel.config.schema import AppConfig
from sentinel.edge_agent.context import EdgeAgentContext
from sentinel.edge_agent.health import build_health_snapshot, write_health_snapshot


def _request():
    return MissionRequest(
        mission_id="health_test",
        launch_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=0),
        area_of_interest=[
            GeoPoint(lat=38.000, lon=-8.000, alt_m=80),
            GeoPoint(lat=38.000, lon=-7.998, alt_m=80),
            GeoPoint(lat=38.002, lon=-7.998, alt_m=80),
        ],
    )


def _context(tmp_path):
    ctx = EdgeAgentContext(
        config=AppConfig(),
        mission_request=_request(),
        event_bus=EventBus(),
        mode="mock",
    )
    ctx.run_dir = tmp_path
    return ctx


def test_health_snapshot_has_mission_id_and_mode(tmp_path):
    ctx = _context(tmp_path)
    snap = build_health_snapshot(ctx)
    assert snap["mission_id"] == "health_test"
    assert snap["mode"] == "mock"


def test_health_snapshot_detects_artifacts(tmp_path):
    ctx = _context(tmp_path)
    (tmp_path / "events.jsonl").write_text('{"test": true}\n', encoding="utf-8")
    (tmp_path / "mission_plan.json").write_text("{}", encoding="utf-8")
    snap = build_health_snapshot(ctx)
    assert snap["artifacts"]["events_jsonl"] is True
    assert snap["artifacts"]["mission_plan_json"] is True
    assert snap["artifacts"]["detections_jsonl"] is False


def test_health_snapshot_includes_vehicle_state(tmp_path):
    ctx = _context(tmp_path)
    ctx.set_vehicle_state(
        VehicleState(
            vehicle_id="uav_001",
            timestamp_utc=utc_now_iso(),
            position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
            mode="SIMULATED",
            armed=False,
            battery_pct=90.0,
        )
    )
    snap = build_health_snapshot(ctx)
    assert snap["vehicle"] is not None
    assert snap["vehicle"]["armed"] is False
    assert snap["vehicle"]["battery_pct"] == 90.0


def test_write_health_snapshot_creates_valid_json(tmp_path):
    ctx = _context(tmp_path)
    output = tmp_path / "edge_agent_health.json"
    write_health_snapshot(ctx, output)
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["mission_id"] == "health_test"
    assert "artifacts" in data
