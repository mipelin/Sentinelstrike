"""Tests for OperatorDecisionGate."""


from sentinel.common.event_bus import EventBus
from sentinel.common.types import GeoObservation, GeoPoint, Track
from sentinel.operator.decisions import OperatorAction, make_decision_record
from sentinel.operator.gate import OperatorDecisionGate
from sentinel.operator.policy import OperatorPolicy


def _track(conf=0.9):
    return Track(
        track_id="trk_001",
        class_name="person",
        confidence=conf,
        last_seen_utc="2026-01-01T00:00:00Z",
    )


def _obs(conf=0.9, obs_id="obs_001"):
    return GeoObservation(
        observation_id=obs_id,
        mission_id="m1",
        track_id="trk_001",
        class_name="person",
        timestamp_utc="2026-01-01T00:00:00Z",
        estimated_location=GeoPoint(lat=38.0, lon=-8.0, alt_m=0.0),
        confidence=conf,
    )


def test_gate_creates_decisions_jsonl(tmp_path):
    bus = EventBus()
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6)
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus, run_dir=tmp_path)

    decisions = gate.evaluate_observations([_obs()], [_track()])
    for d in decisions:
        gate.record_decision(d)

    gate.close()
    assert (tmp_path / "operator_decisions.jsonl").exists()
    lines = (tmp_path / "operator_decisions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_gate_creates_metrics_json(tmp_path):
    bus = EventBus()
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6)
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus, run_dir=tmp_path)

    decisions = gate.evaluate_observations([_obs()], [_track()])
    for d in decisions:
        gate.record_decision(d)

    metrics = gate.close()
    assert (tmp_path / "operator_metrics.json").exists()
    assert metrics["decisions_recorded"] == 1


def test_gate_publishes_events():
    events = []
    bus = EventBus()
    bus.subscribe("*", events.append)
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6)
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus)

    decisions = gate.evaluate_observations([_obs()], [_track()])
    for d in decisions:
        gate.record_decision(d)

    gate.close()

    event_types = {e.event_type for e in events}
    assert "operator_decision_required" in event_types
    assert "operator_decision_recorded" in event_types
    assert "operator_gate_completed" in event_types


def test_gate_confirm_publishes_confirmed_event():
    events = []
    bus = EventBus()
    bus.subscribe("*", events.append)
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6, auto_reject_below_confidence=False, require_human_for_orbit=False)
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus)

    rec = make_decision_record(
        mission_id="m1", operator_id="op1",
        action=OperatorAction.CONFIRM_OBSERVATION,
        observation_id="obs_001", confidence=0.9,
    )
    gate.record_decision(rec)

    event_types = [e.event_type for e in events]
    assert "observation_confirmed" in event_types


def test_gate_reject_publishes_rejected_event():
    events = []
    bus = EventBus()
    bus.subscribe("*", events.append)
    policy = OperatorPolicy()
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus)

    rec = make_decision_record(
        mission_id="m1", operator_id="op1",
        action=OperatorAction.REJECT_OBSERVATION,
        observation_id="obs_001", confidence=0.3,
    )
    gate.record_decision(rec)

    event_types = [e.event_type for e in events]
    assert "observation_rejected" in event_types


def test_gate_abort_sets_flag():
    bus = EventBus()
    policy = OperatorPolicy()
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus)

    assert gate.abort_requested is False
    rec = make_decision_record(
        mission_id="m1", operator_id="op1",
        action=OperatorAction.ABORT_MISSION, reason="safety",
    )
    gate.record_decision(rec)
    assert gate.abort_requested is True


def test_gate_return_to_safe_sets_flag():
    bus = EventBus()
    policy = OperatorPolicy()
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus)

    assert gate.return_to_safe_requested is False
    rec = make_decision_record(
        mission_id="m1", operator_id="op1",
        action=OperatorAction.RETURN_TO_SAFE, reason="loss of link",
    )
    gate.record_decision(rec)
    assert gate.return_to_safe_requested is True


def test_gate_multiple_observations(tmp_path):
    bus = EventBus()
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.5)
    gate = OperatorDecisionGate(mission_id="m1", policy=policy, event_bus=bus, run_dir=tmp_path)

    obs_list = [_obs(conf=0.9, obs_id=f"obs_{i}") for i in range(3)]
    decisions = gate.evaluate_observations(obs_list, [_track(0.8)])
    for d in decisions:
        gate.record_decision(d)

    metrics = gate.close()
    assert metrics["decisions_recorded"] == 3
