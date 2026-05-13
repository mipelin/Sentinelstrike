"""Tests for safety metrics computation."""

from sentinel.safety.actions import SafetyAction, SafetyTrigger, make_safety_action_record
from sentinel.safety.metrics import compute_safety_metrics


def test_empty_metrics():
    m = compute_safety_metrics([])
    assert m["total_actions"] == 0
    assert m["successful_actions"] == 0
    assert m["failed_actions"] == 0


def test_metrics_with_records():
    records = [
        make_safety_action_record("m1", SafetyTrigger.OPERATOR_ABORT, SafetyAction.HOLD, command_success=True),
        make_safety_action_record("m1", SafetyTrigger.OPERATOR_ABORT, SafetyAction.LAND, command_success=True),
    ]
    m = compute_safety_metrics(records)
    assert m["total_actions"] == 2
    assert m["successful_actions"] == 2
    assert m["failed_actions"] == 0
    assert "OPERATOR_ABORT" in m["triggers_fired"]
    assert "HOLD" in m["actions_executed"]
    assert "LAND" in m["actions_executed"]


def test_metrics_with_failure():
    records = [
        make_safety_action_record("m1", SafetyTrigger.LOW_BATTERY, SafetyAction.RETURN_HOME, command_success=False, command_message="rejected"),
    ]
    m = compute_safety_metrics(records)
    assert m["total_actions"] == 1
    assert m["failed_actions"] == 1
    assert m["successful_actions"] == 0


def test_action_counts():
    records = [
        make_safety_action_record("m1", SafetyTrigger.OPERATOR_ABORT, SafetyAction.HOLD, command_success=True),
        make_safety_action_record("m1", SafetyTrigger.OPERATOR_ABORT, SafetyAction.LAND, command_success=True),
        make_safety_action_record("m1", SafetyTrigger.LINK_LOSS, SafetyAction.RETURN_HOME, command_success=True),
    ]
    m = compute_safety_metrics(records)
    assert m["action_counts"]["HOLD"] == 1
    assert m["action_counts"]["LAND"] == 1
    assert m["action_counts"]["RETURN_HOME"] == 1
