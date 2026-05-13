"""Mission plan metrics calculation."""

from __future__ import annotations

from sentinel.common.geo import path_distance_m
from sentinel.common.types import GeoPoint


def estimate_duration_s(distance_m: float, speed_mps: float) -> float:
    if speed_mps <= 0:
        raise ValueError(f"speed_mps must be positive, got {speed_mps}")
    return distance_m / speed_mps


def plan_metrics(waypoints: list[GeoPoint], speed_mps: float) -> dict:
    dist = path_distance_m(waypoints)
    duration = estimate_duration_s(dist, speed_mps)
    return {
        "total_distance_m": dist,
        "estimated_duration_s": duration,
        "waypoint_count": len(waypoints),
    }
