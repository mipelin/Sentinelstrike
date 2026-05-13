"""Safety action types, triggers, and action records."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from sentinel.common.time import utc_now_iso


class SafetyAction(StrEnum):
    HOLD = "HOLD"
    RETURN_HOME = "RETURN_HOME"
    LAND = "LAND"
    DISARM = "DISARM"
    NO_ACTION = "NO_ACTION"


class SafetyTrigger(StrEnum):
    OPERATOR_ABORT = "OPERATOR_ABORT"
    OPERATOR_RETURN_TO_SAFE = "OPERATOR_RETURN_TO_SAFE"
    LINK_LOSS = "LINK_LOSS"
    LOW_BATTERY = "LOW_BATTERY"
    TELEMETRY_STALE = "TELEMETRY_STALE"
    TELEMETRY_INVALID = "TELEMETRY_INVALID"
    MANUAL = "MANUAL"


class SafetyActionRecord(BaseModel):
    record_id: str
    mission_id: str
    timestamp_utc: str
    trigger: SafetyTrigger
    action: SafetyAction
    command_success: bool
    command_message: str = ""
    metadata: dict = {}


def _make_record_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def make_safety_action_record(
    mission_id: str,
    trigger: SafetyTrigger,
    action: SafetyAction,
    *,
    command_success: bool = True,
    command_message: str = "",
    metadata: dict | None = None,
) -> SafetyActionRecord:
    return SafetyActionRecord(
        record_id=_make_record_id(),
        mission_id=mission_id,
        timestamp_utc=utc_now_iso(),
        trigger=trigger,
        action=action,
        command_success=command_success,
        command_message=command_message,
        metadata=metadata or {},
    )
