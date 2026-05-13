"""Tests for operator policy evaluation."""

from sentinel.common.types import BoundingBox, GeoObservation, GeoPoint, Track
from sentinel.operator.decisions import OperatorAction
from sentinel.operator.policy import OperatorPolicy


def _track(conf=0.8, cls="person"):
    return Track(
        track_id="trk_001",
        class_name=cls,
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=10, y1=10, x2=100, y2=100),
        last_seen_utc="2026-01-01T00:00:00Z",
    )


def _obs(conf=0.8):
    return GeoObservation(
        observation_id="obs_001",
        mission_id="m1",
        track_id="trk_001",
        class_name="person",
        timestamp_utc="2026-01-01T00:00:00Z",
        estimated_location=GeoPoint(lat=38.0, lon=-8.0, alt_m=0.0),
        confidence=conf,
    )


def test_high_confidence_confirms():
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6, auto_reject_below_confidence=True)
    result = policy.evaluate_observation(track=_track(0.9), observation=_obs(0.9))
    assert result.allowed is True
    assert result.required_action == OperatorAction.CONFIRM_OBSERVATION


def test_low_track_confidence_rejects():
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6, auto_reject_below_confidence=True)
    result = policy.evaluate_observation(track=_track(0.3), observation=_obs(0.9))
    assert result.allowed is False
    assert result.required_action == OperatorAction.REJECT_OBSERVATION


def test_low_obs_confidence_rejects():
    policy = OperatorPolicy(min_observation_confidence_for_orbit=0.6, auto_reject_below_confidence=True)
    result = policy.evaluate_observation(track=_track(0.9), observation=_obs(0.3))
    assert result.allowed is False
    assert result.required_action == OperatorAction.REJECT_OBSERVATION


def test_auto_reject_disabled():
    policy = OperatorPolicy(
        min_track_confidence_for_confirmation=0.6,
        auto_reject_below_confidence=False,
    )
    result = policy.evaluate_observation(track=_track(0.3), observation=_obs(0.9))
    # With auto_reject off, low confidence doesn't auto-reject
    assert result.required_action != OperatorAction.REJECT_OBSERVATION or result.allowed is True


def test_can_auto_publish():
    policy = OperatorPolicy(min_track_confidence_for_confirmation=0.6, min_observation_confidence_for_orbit=0.6)
    assert policy.can_auto_publish(track=_track(0.9), observation=_obs(0.9)) is True
    assert policy.can_auto_publish(track=_track(0.3), observation=_obs(0.9)) is False


def test_requires_operator_decision():
    policy = OperatorPolicy(
        min_track_confidence_for_confirmation=0.6,
        min_observation_confidence_for_orbit=0.6,
        require_human_for_orbit=True,
    )
    # Above threshold but require_human_for_orbit=True → still requires human
    assert policy.requires_operator_decision(track=_track(0.9), observation=_obs(0.9)) is True


def test_no_require_human_for_orbit():
    policy = OperatorPolicy(
        min_track_confidence_for_confirmation=0.6,
        min_observation_confidence_for_orbit=0.6,
        require_human_for_orbit=False,
    )
    # Above threshold and no human required → auto
    assert policy.requires_operator_decision(track=_track(0.9), observation=_obs(0.9)) is False
