"""Tests for geolocalizer metrics."""

from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoObservation, GeoPoint
from sentinel.geolocalizer.metrics import compute_geolocalizer_metrics


def _obs(class_name="person", conf=0.9, acc=35.0, method="flat_ground_pinhole_v1"):
    return GeoObservation(
        observation_id="obs_001",
        mission_id="m1",
        track_id="trk_000001",
        class_name=class_name,
        timestamp_utc=utc_now_iso(),
        estimated_location=GeoPoint(lat=38.0, lon=-8.0, alt_m=0.0),
        accuracy_estimate_m=acc,
        method=method,
        confidence=conf,
    )


def test_counts_observations():
    obs = [_obs(), _obs(class_name="car")]
    m = compute_geolocalizer_metrics(obs)
    assert m["observation_count"] == 2
    assert m["classes"] == {"person": 1, "car": 1}


def test_average_confidence():
    obs = [_obs(conf=0.8), _obs(conf=0.6)]
    m = compute_geolocalizer_metrics(obs)
    assert abs(m["average_confidence"] - 0.7) < 1e-6


def test_no_observations():
    m = compute_geolocalizer_metrics([])
    assert m["observation_count"] == 0
    assert m["average_confidence"] == 0.0
    assert m["average_accuracy_estimate_m"] == 0.0
