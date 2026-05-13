"""Tests for dashboard timeline builder."""

from sentinel.dashboard.timeline import build_timeline


def test_empty_artifacts():
    events = build_timeline({})
    assert events == []


def test_system_events():
    artifacts = {
        "events": [
            {"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "test_event", "severity": "info"},
            {"timestamp_utc": "2026-01-01T00:00:02Z", "event_type": "pipeline_completed", "severity": "info"},
        ],
    }
    events = build_timeline(artifacts)
    assert len(events) == 2
    assert events[0].source == "system"
    assert events[0].type == "test_event"


def test_operator_decisions():
    artifacts = {
        "operator_decisions": [
            {"timestamp_utc": "2026-01-01T00:01:00Z", "action": "CONFIRM_OBSERVATION", "observation_id": "obs_1"},
            {"timestamp_utc": "2026-01-01T00:02:00Z", "action": "ABORT_MISSION", "reason": "safety"},
        ],
    }
    events = build_timeline(artifacts)
    assert len(events) == 2
    assert events[0].source == "operator"
    assert events[1].severity == "warning"
    assert "ABORT" in events[1].label


def test_safety_actions():
    artifacts = {
        "safety_actions": [
            {"timestamp_utc": "2026-01-01T00:03:00Z", "action": "LAND", "trigger": "OPERATOR_ABORT"},
        ],
    }
    events = build_timeline(artifacts)
    assert len(events) == 1
    assert events[0].source == "safety"
    assert events[0].severity == "error"


def test_geo_observations():
    artifacts = {
        "geo_observations": [
            {"timestamp_utc": "2026-01-01T00:00:05Z", "class_name": "person", "confidence": 0.85},
        ],
    }
    events = build_timeline(artifacts)
    assert len(events) == 1
    assert events[0].source == "geo"
    assert "person" in events[0].label


def test_timeline_sorted_by_timestamp():
    artifacts = {
        "events": [
            {"timestamp_utc": "2026-01-01T00:00:02Z", "event_type": "late"},
            {"timestamp_utc": "2026-01-01T00:00:00Z", "event_type": "early"},
        ],
        "safety_actions": [
            {"timestamp_utc": "2026-01-01T00:00:01Z", "action": "HOLD", "trigger": "MANUAL"},
        ],
    }
    events = build_timeline(artifacts)
    assert len(events) == 3
    assert events[0].type == "early"
    assert events[1].source == "safety"
    assert events[2].type == "late"


def test_missing_timestamp_at_end():
    artifacts = {
        "events": [
            {"timestamp_utc": "2026-01-01T00:00:01Z", "event_type": "with_ts"},
            {"event_type": "no_ts"},
        ],
    }
    events = build_timeline(artifacts)
    assert len(events) == 2
    assert events[0].type == "with_ts"
    assert events[1].type == "no_ts"
