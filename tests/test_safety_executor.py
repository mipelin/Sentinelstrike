"""Tests for SafetyActionExecutor."""

from pathlib import Path

from sentinel.common.event_bus import EventBus
from sentinel.common.types import GeoPoint, VehicleState
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
from sentinel.safety.actions import SafetyAction, SafetyTrigger
from sentinel.safety.executor import SafetyActionExecutor
from sentinel.safety.policy import SafetyPolicyConfig, SafetyPolicyV1


def _make_executor(tmp_path: Path | None = None, policy: SafetyPolicyV1 | None = None) -> tuple[SafetyActionExecutor, MavlinkBridge]:
    backend = MockMavlinkBackend(vehicle_id="uav_001")
    bus = EventBus()
    bridge = MavlinkBridge(mission_id="m1", vehicle_id="uav_001", backend=backend, event_bus=bus)
    p = policy or SafetyPolicyV1()
    executor = SafetyActionExecutor(
        mission_id="m1", policy=p, mavlink_bridge=bridge,
        event_bus=bus, run_dir=tmp_path,
    )
    return executor, bridge


def _vehicle(battery: float = 80.0) -> VehicleState:
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc="2026-01-01T00:00:00Z",
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=50.0),
        battery_pct=battery,
    )


def test_abort_triggers_hold_then_land(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs = executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    assert len(recs) == 2
    assert recs[0].action == SafetyAction.HOLD
    assert recs[1].action == SafetyAction.LAND
    assert all(r.command_success for r in recs)
    assert executor.abort_executed is True


def test_return_to_safe_triggers_rth(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs = executor.evaluate_operator_decision(abort_requested=False, return_to_safe_requested=True)
    assert len(recs) == 1
    assert recs[0].action == SafetyAction.RETURN_HOME
    assert executor.return_to_safe_executed is True


def test_one_shot_abort(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs1 = executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    assert len(recs1) == 2
    recs2 = executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    assert len(recs2) == 0


def test_reset_rearms_triggers(tmp_path):
    executor, _ = _make_executor(tmp_path)
    executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    executor.reset()
    recs = executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    assert len(recs) == 2


def test_low_battery_trigger(tmp_path):
    policy = SafetyPolicyV1(SafetyPolicyConfig(low_battery_threshold_pct=30.0))
    executor, _ = _make_executor(tmp_path, policy)
    recs = executor.evaluate_system_conditions(vehicle_state=_vehicle(battery=15.0))
    assert len(recs) == 1
    assert recs[0].trigger == SafetyTrigger.LOW_BATTERY
    assert recs[0].action == SafetyAction.RETURN_HOME


def test_no_battery_trigger_above_threshold(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs = executor.evaluate_system_conditions(vehicle_state=_vehicle(battery=80.0))
    assert len(recs) == 0


def test_telemetry_invalid_trigger(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs = executor.evaluate_system_conditions(vehicle_state=None)
    assert len(recs) == 1
    assert recs[0].trigger == SafetyTrigger.TELEMETRY_INVALID


def test_telemetry_stale_trigger(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs = executor.evaluate_system_conditions(vehicle_state=_vehicle(), stale_count=5)
    assert len(recs) == 1
    assert recs[0].trigger == SafetyTrigger.TELEMETRY_STALE


def test_link_loss_trigger(tmp_path):
    executor, _ = _make_executor(tmp_path)
    recs = executor.evaluate_system_conditions(vehicle_state=_vehicle(), link_lost=True)
    assert len(recs) == 1
    assert recs[0].trigger == SafetyTrigger.LINK_LOSS


def test_creates_safety_actions_jsonl(tmp_path):
    executor, _ = _make_executor(tmp_path)
    executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    executor.close()
    assert (tmp_path / "safety_actions.jsonl").exists()


def test_creates_safety_metrics_json(tmp_path):
    executor, _ = _make_executor(tmp_path)
    executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    metrics = executor.close()
    assert (tmp_path / "safety_metrics.json").exists()
    assert metrics["total_actions"] == 2
    assert metrics["successful_actions"] == 2


def test_publishes_events(tmp_path):
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)
    backend = MockMavlinkBackend(vehicle_id="uav_001")
    bridge = MavlinkBridge(mission_id="m1", vehicle_id="uav_001", backend=backend, event_bus=bus)
    executor = SafetyActionExecutor(mission_id="m1", policy=SafetyPolicyV1(), mavlink_bridge=bridge, event_bus=bus)

    executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)

    event_types = [e.event_type for e in events]
    assert "safety_action_executed" in event_types


def test_fire_manual(tmp_path):
    executor, _ = _make_executor(tmp_path)
    rec = executor.fire_manual(SafetyAction.LAND)
    assert rec.action == SafetyAction.LAND
    assert rec.trigger == SafetyTrigger.MANUAL
    assert rec.command_success is True


def test_disarm_blocked_by_default(tmp_path):
    policy = SafetyPolicyV1(SafetyPolicyConfig(abort_sequence=[SafetyAction.DISARM]))
    executor, _ = _make_executor(tmp_path, policy)
    recs = executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
    assert len(recs) == 1
    assert recs[0].command_success is False
    assert "blocked" in recs[0].command_message.lower()
