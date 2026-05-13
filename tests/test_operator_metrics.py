"""Tests for operator metrics computation."""

from sentinel.operator.decisions import OperatorAction, make_decision_record
from sentinel.operator.metrics import compute_operator_metrics


def test_empty_metrics():
    m = compute_operator_metrics([])
    assert m["decisions_required"] == 0
    assert m["confirmed_count"] == 0
    assert m["rejected_count"] == 0
    assert m["average_decision_confidence"] == 0.0


def test_confirmed_metrics():
    decisions = [
        make_decision_record("m1", "op1", OperatorAction.CONFIRM_OBSERVATION, confidence=0.9),
        make_decision_record("m1", "op1", OperatorAction.CONFIRM_OBSERVATION, confidence=0.8),
    ]
    m = compute_operator_metrics(decisions)
    assert m["confirmed_count"] == 2
    assert m["rejected_count"] == 0
    assert abs(m["average_decision_confidence"] - 0.85) < 0.01


def test_mixed_metrics():
    decisions = [
        make_decision_record("m1", "op1", OperatorAction.CONFIRM_OBSERVATION, confidence=0.9),
        make_decision_record("m1", "op1", OperatorAction.REJECT_OBSERVATION, confidence=0.3),
        make_decision_record("m1", "op1", OperatorAction.ABORT_MISSION, reason="safety"),
    ]
    m = compute_operator_metrics(decisions)
    assert m["decisions_recorded"] == 3
    assert m["confirmed_count"] == 1
    assert m["rejected_count"] == 1
    assert m["abort_count"] == 1


def test_return_to_safe_count():
    decisions = [
        make_decision_record("m1", "op1", OperatorAction.RETURN_TO_SAFE, reason="link loss"),
    ]
    m = compute_operator_metrics(decisions)
    assert m["return_to_safe_count"] == 1
