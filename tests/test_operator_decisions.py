"""Tests for operator decision models."""

from sentinel.operator.decisions import (
    OperatorAction,
    OperatorDecisionRecord,
    PolicyDecision,
    make_decision_record,
)


def test_operator_action_values():
    assert OperatorAction.CONFIRM_OBSERVATION == "CONFIRM_OBSERVATION"
    assert OperatorAction.REJECT_OBSERVATION == "REJECT_OBSERVATION"
    assert OperatorAction.ABORT_MISSION == "ABORT_MISSION"
    assert OperatorAction.RETURN_TO_SAFE == "RETURN_TO_SAFE"
    assert OperatorAction.NO_ACTION == "NO_ACTION"


def test_policy_decision_model():
    pd = PolicyDecision(allowed=True, required_action=OperatorAction.CONFIRM_OBSERVATION, reason="ok")
    assert pd.allowed is True
    assert pd.severity == "info"


def test_make_decision_record():
    rec = make_decision_record(
        mission_id="m1",
        operator_id="op1",
        action=OperatorAction.CONFIRM_OBSERVATION,
        track_id="trk_001",
        observation_id="obs_001",
        reason="high confidence",
        confidence=0.9,
    )
    assert rec.mission_id == "m1"
    assert rec.operator_id == "op1"
    assert rec.action == OperatorAction.CONFIRM_OBSERVATION
    assert rec.track_id == "trk_001"
    assert rec.confidence_at_decision == 0.9
    assert rec.source == "operator_gate"
    assert len(rec.decision_id) > 0
    assert len(rec.timestamp_utc) > 0


def test_decision_record_serialization():
    rec = make_decision_record(
        mission_id="m1",
        operator_id="op1",
        action=OperatorAction.REJECT_OBSERVATION,
        confidence=0.3,
    )
    json_str = rec.model_dump_json()
    parsed = OperatorDecisionRecord.model_validate_json(json_str)
    assert parsed.action == OperatorAction.REJECT_OBSERVATION
    assert parsed.confidence_at_decision == 0.3
