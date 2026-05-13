"""Timeline builder — combines events from multiple artifact sources."""

from __future__ import annotations

from .models import TimelineEvent


def build_timeline(artifacts: dict) -> list[TimelineEvent]:
    """Build a unified, sorted timeline from all artifacts."""
    events: list[TimelineEvent] = []

    _add_system_events(events, artifacts.get("events", []))
    _add_operator_decisions(events, artifacts.get("operator_decisions", []))
    _add_safety_actions(events, artifacts.get("safety_actions", []))
    _add_geo_observations(events, artifacts.get("geo_observations", []))

    events.sort(key=lambda e: (e.t == "", e.t))
    return events


def _add_system_events(events: list[TimelineEvent], items: list[dict]) -> None:
    for item in items:
        events.append(
            TimelineEvent(
                t=item.get("timestamp_utc", ""),
                source="system",
                type=item.get("event_type", "unknown"),
                severity=item.get("severity", "info"),
                label=item.get("event_type", "system event"),
                payload=item,
            )
        )


def _add_operator_decisions(events: list[TimelineEvent], items: list[dict]) -> None:
    for item in items:
        action = item.get("action", "UNKNOWN")
        severity = "warning" if action in ("ABORT_MISSION", "RETURN_TO_SAFE") else "info"
        events.append(
            TimelineEvent(
                t=item.get("timestamp_utc", ""),
                source="operator",
                type=f"operator_{action.lower()}",
                severity=severity,
                label=f"Operator: {action}",
                payload=item,
            )
        )


def _add_safety_actions(events: list[TimelineEvent], items: list[dict]) -> None:
    for item in items:
        action = item.get("action", "UNKNOWN")
        trigger = item.get("trigger", "UNKNOWN")
        severity = "error" if action in ("LAND", "DISARM") else "warning"
        events.append(
            TimelineEvent(
                t=item.get("timestamp_utc", ""),
                source="safety",
                type=f"safety_{action.lower()}",
                severity=severity,
                label=f"Safety: {action} ({trigger})",
                payload=item,
            )
        )


def _add_geo_observations(events: list[TimelineEvent], items: list[dict]) -> None:
    for item in items:
        cls_name = item.get("class_name", "unknown")
        conf = item.get("confidence", 0)
        events.append(
            TimelineEvent(
                t=item.get("timestamp_utc", ""),
                source="geo",
                type="geo_observation",
                severity="info",
                label=f"Detected: {cls_name} ({conf:.0%})",
                payload=item,
            )
        )
