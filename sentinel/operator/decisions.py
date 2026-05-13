"""Operator decision models — Pydantic types for the decision gate."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from sentinel.common.time import utc_now_iso


class OperatorAction(StrEnum):
    CONFIRM_OBSERVATION = "CONFIRM_OBSERVATION"
    REJECT_OBSERVATION = "REJECT_OBSERVATION"
    REQUEST_REACQUIRE = "REQUEST_REACQUIRE"
    AUTHORIZE_OBSERVE_ORBIT = "AUTHORIZE_OBSERVE_ORBIT"
    ABORT_MISSION = "ABORT_MISSION"
    RETURN_TO_SAFE = "RETURN_TO_SAFE"
    NO_ACTION = "NO_ACTION"


class OperatorDecisionRecord(BaseModel):
    decision_id: str
    mission_id: str
    timestamp_utc: str
    operator_id: str
    action: OperatorAction
    track_id: str | None = None
    observation_id: str | None = None
    reason: str = ""
    confidence_at_decision: float = 0.0
    source: str = "operator_gate"
    metadata: dict = {}


class PolicyDecision(BaseModel):
    allowed: bool
    required_action: OperatorAction
    reason: str = ""
    severity: str = "info"


def make_decision_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def make_decision_record(
    mission_id: str,
    operator_id: str,
    action: OperatorAction,
    *,
    track_id: str | None = None,
    observation_id: str | None = None,
    reason: str = "",
    confidence: float = 0.0,
    source: str = "operator_gate",
    metadata: dict | None = None,
) -> OperatorDecisionRecord:
    return OperatorDecisionRecord(
        decision_id=make_decision_id(),
        mission_id=mission_id,
        timestamp_utc=utc_now_iso(),
        operator_id=operator_id,
        action=action,
        track_id=track_id,
        observation_id=observation_id,
        reason=reason,
        confidence_at_decision=confidence,
        source=source,
        metadata=metadata or {},
    )
