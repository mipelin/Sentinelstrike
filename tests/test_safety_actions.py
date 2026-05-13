"""Tests for safety action types and records."""

from sentinel.safety.actions import (
    SafetyAction,
    SafetyTrigger,
    make_safety_action_record,
)


def test_safety_action_values():
    assert SafetyAction.HOLD == "HOLD"
    assert SafetyAction.RETURN_HOME == "RETURN_HOME"
    assert SafetyAction.LAND == "LAND"
    assert SafetyAction.DISARM == "DISARM"
    assert SafetyAction.NO_ACTION == "NO_ACTION"


def test_safety_trigger_values():
    assert SafetyTrigger.OPERATOR_ABORT == "OPERATOR_ABORT"
    assert SafetyTrigger.LINK_LOSS == "LINK_LOSS"
    assert SafetyTrigger.LOW_BATTERY == "LOW_BATTERY"
    assert SafetyTrigger.TELEMETRY_STALE == "TELEMETRY_STALE"


def test_make_safety_action_record():
    rec = make_safety_action_record(
        mission_id="m1",
        trigger=SafetyTrigger.OPERATOR_ABORT,
        action=SafetyAction.LAND,
        command_success=True,
        command_message="landed safely",
    )
    assert rec.mission_id == "m1"
    assert rec.trigger == SafetyTrigger.OPERATOR_ABORT
    assert rec.action == SafetyAction.LAND
    assert rec.command_success is True
    assert rec.command_message == "landed safely"
    assert rec.record_id
    assert rec.timestamp_utc


def test_record_with_metadata():
    rec = make_safety_action_record(
        mission_id="m1",
        trigger=SafetyTrigger.LOW_BATTERY,
        action=SafetyAction.RETURN_HOME,
        metadata={"battery_pct": 12.5},
    )
    assert rec.metadata["battery_pct"] == 12.5
