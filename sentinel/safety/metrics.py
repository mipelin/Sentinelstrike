"""Safety metrics computation."""

from __future__ import annotations

from .actions import SafetyActionRecord


def compute_safety_metrics(records: list[SafetyActionRecord]) -> dict:
    """Compute summary metrics from safety action records."""
    if not records:
        return {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "triggers_fired": [],
            "actions_executed": [],
        }

    successful = sum(1 for r in records if r.command_success)
    failed = len(records) - successful
    triggers = list({str(r.trigger) for r in records})
    actions = [str(r.action) for r in records]

    action_counts: dict[str, int] = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    return {
        "total_actions": len(records),
        "successful_actions": successful,
        "failed_actions": failed,
        "triggers_fired": triggers,
        "actions_executed": actions,
        "action_counts": action_counts,
    }
