"""Tests for mission metrics."""

import pytest

from sentinel.common.types import GeoPoint
from sentinel.mission_planner.metrics import estimate_duration_s, plan_metrics


def test_path_distance_positive():
    points = [
        GeoPoint(lat=38.0, lon=-8.0),
        GeoPoint(lat=38.001, lon=-8.0),
        GeoPoint(lat=38.001, lon=-7.999),
    ]
    metrics = plan_metrics(points, speed_mps=10)
    assert metrics["total_distance_m"] > 0


def test_estimate_duration():
    assert estimate_duration_s(1000, 10) == 100.0


def test_estimate_duration_zero_speed():
    with pytest.raises(ValueError, match="positive"):
        estimate_duration_s(1000, 0)


def test_plan_metrics_keys():
    points = [GeoPoint(lat=38, lon=-8), GeoPoint(lat=38.001, lon=-8)]
    m = plan_metrics(points, speed_mps=10)
    assert "total_distance_m" in m
    assert "estimated_duration_s" in m
    assert "waypoint_count" in m
    assert m["waypoint_count"] == 2
