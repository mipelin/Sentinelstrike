"""Tests for OperatorSimulator."""

from sentinel.common.types import GeoObservation, GeoPoint, Track
from sentinel.operator.decisions import OperatorAction
from sentinel.operator.policy import OperatorPolicy
from sentinel.operator.simulator import OperatorSimulator


def _track(conf=0.9):
    return Track(
        track_id="trk_001",
        class_name="person",
        confidence=conf,
        last_seen_utc="2026-01-01T00:00:00Z",
    )


def _obs(conf=0.9):
    return GeoObservation(
        observation_id="obs_001",
        mission_id="m1",
        track_id="trk_001",
        class_name="person",
        timestamp_utc="2026-01-01T00:00:00Z",
        estimated_location=GeoPoint(lat=38.0, lon=-8.0, alt_m=0.0),
        confidence=conf,
    )


def test_auto_confirm():
    sim = OperatorSimulator(mode="auto_confirm")
    rec = sim.generate_decision(_obs(), _track(), "m1")
    assert rec is not None
    assert rec.action == OperatorAction.CONFIRM_OBSERVATION


def test_auto_reject():
    sim = OperatorSimulator(mode="auto_reject")
    rec = sim.generate_decision(_obs(), _track(), "m1")
    assert rec is not None
    assert rec.action == OperatorAction.REJECT_OBSERVATION


def test_confirm_above_threshold_high_confidence():
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6, min_observation_confidence_for_orbit=0.6)
    sim = OperatorSimulator(mode="confirm_above_threshold", policy=policy)
    rec = sim.generate_decision(_obs(0.9), _track(0.9), "m1")
    assert rec is not None
    assert rec.action == OperatorAction.CONFIRM_OBSERVATION


def test_confirm_above_threshold_low_confidence():
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6, min_observation_confidence_for_orbit=0.6)
    sim = OperatorSimulator(mode="confirm_above_threshold", policy=policy)
    rec = sim.generate_decision(_obs(0.3), _track(0.3), "m1")
    assert rec is not None
    assert rec.action == OperatorAction.REJECT_OBSERVATION


def test_no_action():
    sim = OperatorSimulator(mode="no_action")
    rec = sim.generate_decision(_obs(), _track(), "m1")
    assert rec is None


def test_simulator_sets_source():
    sim = OperatorSimulator(mode="auto_confirm", operator_id="test_op")
    rec = sim.generate_decision(_obs(), _track(), "m1")
    assert rec.source == "operator_simulator"
    assert rec.operator_id == "test_op"


def test_mode_property():
    sim = OperatorSimulator(mode="auto_confirm")
    assert sim.mode == "auto_confirm"
