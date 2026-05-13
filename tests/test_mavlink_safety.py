"""Tests for MAVLink safety policy."""

import pytest

from sentinel.common.time import utc_now_iso
from sentinel.mavlink_bridge.commands import MavlinkCommand
from sentinel.mavlink_bridge.safety import MavlinkSafetyConfig, SafetyPolicy


def _cmd(command_type: str, **payload: object) -> MavlinkCommand:
    return MavlinkCommand(
        command_id="test",
        mission_id="test",
        command_type=command_type,
        timestamp_utc=utc_now_iso(),
        payload=payload,
    )


def _policy(**overrides: object) -> SafetyPolicy:
    cfg = MavlinkSafetyConfig(**overrides)
    return SafetyPolicy(cfg, "SIMULATION_MODE")


def test_arm_blocked_by_default():
    policy = _policy()
    with pytest.raises(ValueError, match="ARM"):
        policy.validate_command(_cmd("ARM"))


def test_takeoff_blocked_by_default():
    policy = _policy()
    with pytest.raises(ValueError, match="TAKEOFF"):
        policy.validate_command(_cmd("TAKEOFF"))


def test_takeoff_blocked_if_altitude_exceeds_max():
    policy = _policy(allow_takeoff=True, max_takeoff_altitude_m=30)
    with pytest.raises(ValueError, match="exceeds maximum"):
        policy.validate_command(_cmd("TAKEOFF", altitude_m=50))


def test_real_backend_blocked_by_default():
    policy = _policy()
    with pytest.raises(ValueError, match="Real backend"):
        policy.check_real_backend_allowed()


def test_safe_commands_not_blocked():
    policy = _policy()
    for ct in ("HOLD", "LAND", "RETURN_HOME", "DISARM", "CONNECT", "GET_TELEMETRY"):
        policy.validate_command(_cmd(ct))


def test_simulation_mode_required():
    cfg = MavlinkSafetyConfig(require_simulation_mode=True)
    policy = SafetyPolicy(cfg, "EDGE_MODE")
    with pytest.raises(ValueError, match="SIMULATION_MODE"):
        policy.validate_command(_cmd("ARM", allow_arm=True))


def test_arm_allowed_when_configured():
    policy = _policy(allow_arm=True)
    policy.validate_command(_cmd("ARM"))
