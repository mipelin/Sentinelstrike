"""Operator gate metrics computation."""

from __future__ import annotations

from .decisions import OperatorAction, OperatorDecisionRecord


def compute_operator_metrics(decisions: list[OperatorDecisionRecord]) -> dict:
    total = len(decisions)
    confirmed = sum(1 for d in decisions if d.action == OperatorAction.CONFIRM_OBSERVATION)
    rejected = sum(1 for d in decisions if d.action == OperatorAction.REJECT_OBSERVATION)
    abort = sum(1 for d in decisions if d.action == OperatorAction.ABORT_MISSION)
    return_safe = sum(1 for d in decisions if d.action == OperatorAction.RETURN_TO_SAFE)
    reacquire = sum(1 for d in decisions if d.action == OperatorAction.REQUEST_REACQUIRE)
    orbit = sum(1 for d in decisions if d.action == OperatorAction.AUTHORIZE_OBSERVE_ORBIT)

    confidences = [d.confidence_at_decision for d in decisions if d.confidence_at_decision > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "decisions_required": total,
        "decisions_recorded": total,
        "confirmed_count": confirmed,
        "rejected_count": rejected,
        "abort_count": abort,
        "return_to_safe_count": return_safe,
        "reacquire_count": reacquire,
        "orbit_count": orbit,
        "average_decision_confidence": round(avg_conf, 3),
    }
