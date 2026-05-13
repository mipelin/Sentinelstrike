"""Tests for safety policy evaluation."""

from sentinel.safety.actions import SafetyAction
from sentinel.safety.policy import SafetyPolicyConfig, SafetyPolicyV1


def test_default_abort_sequence():
    policy = SafetyPolicyV1()
    actions = policy.resolve_action("OPERATOR_ABORT")
    assert actions == [SafetyAction.HOLD, SafetyAction.LAND]


def test_return_to_safe_action():
    policy = SafetyPolicyV1()
    action = policy.resolve_action("OPERATOR_RETURN_TO_SAFE")
    assert action == SafetyAction.RETURN_HOME


def test_link_loss_action():
    policy = SafetyPolicyV1()
    action = policy.resolve_action("LINK_LOSS")
    assert action == SafetyAction.RETURN_HOME


def test_low_battery_default():
    policy = SafetyPolicyV1()
    action = policy.resolve_action("LOW_BATTERY")
    assert action == SafetyAction.RETURN_HOME


def test_telemetry_stale_default():
    policy = SafetyPolicyV1()
    action = policy.resolve_action("TELEMETRY_STALE")
    assert action == SafetyAction.HOLD


def test_telemetry_invalid():
    policy = SafetyPolicyV1()
    action = policy.resolve_action("TELEMETRY_INVALID")
    assert action == SafetyAction.HOLD


def test_is_action_allowed():
    policy = SafetyPolicyV1()
    assert policy.is_action_allowed(SafetyAction.HOLD) is True
    assert policy.is_action_allowed(SafetyAction.RETURN_HOME) is True
    assert policy.is_action_allowed(SafetyAction.LAND) is True
    assert policy.is_action_allowed(SafetyAction.DISARM) is False
    assert policy.is_action_allowed(SafetyAction.NO_ACTION) is True


def test_disarm_allowed():
    config = SafetyPolicyConfig(allow_disarm=True)
    policy = SafetyPolicyV1(config)
    assert policy.is_action_allowed(SafetyAction.DISARM) is True


def test_check_battery_below_threshold():
    policy = SafetyPolicyV1()
    assert policy.check_battery(15.0) is True
    assert policy.check_battery(25.0) is False
    assert policy.check_battery(None) is False


def test_check_battery_custom_threshold():
    config = SafetyPolicyConfig(low_battery_threshold_pct=30.0)
    policy = SafetyPolicyV1(config)
    assert policy.check_battery(25.0) is True
    assert policy.check_battery(35.0) is False


def test_check_telemetry_stale():
    policy = SafetyPolicyV1()
    assert policy.check_telemetry_stale(5) is True
    assert policy.check_telemetry_stale(4) is False
    assert policy.check_telemetry_stale(10) is True


def test_custom_abort_sequence():
    config = SafetyPolicyConfig(abort_sequence=[SafetyAction.RETURN_HOME])
    policy = SafetyPolicyV1(config)
    actions = policy.resolve_action("OPERATOR_ABORT")
    assert actions == [SafetyAction.RETURN_HOME]
