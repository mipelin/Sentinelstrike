"""Validation for mission areas and plans."""

from __future__ import annotations

from sentinel.common.geo import bounding_box, haversine_distance_m
from sentinel.common.types import GeoPoint, MissionConstraints, MissionPlan


def validate_area_of_interest(points: list[GeoPoint]) -> None:
    if len(points) < 3:
        raise ValueError("area_of_interest must have at least 3 points")

    first = points[0]
    if all(p.lat == first.lat and p.lon == first.lon for p in points):
        raise ValueError("All points in area_of_interest are identical")

    bbox = bounding_box(points)
    if bbox["min_lat"] == bbox["max_lat"] and bbox["min_lon"] == bbox["max_lon"]:
        raise ValueError("area_of_interest has zero bounding box")


def validate_plan_against_constraints(plan: MissionPlan, constraints: MissionConstraints) -> None:
    effective_max = constraints.max_waypoints + 2  # launch + return
    if len(plan.waypoints) > effective_max:
        raise ValueError(
            f"Plan has {len(plan.waypoints)} waypoints, "
            f"exceeding effective limit of {effective_max} "
            f"(max_waypoints={constraints.max_waypoints} + 2 for launch/return)"
        )

    if plan.total_distance_m is not None:
        max_m = constraints.max_range_km * 1000
        if plan.total_distance_m > max_m:
            raise ValueError(
                f"Plan total distance {plan.total_distance_m:.0f}m "
                f"exceeds max range {max_m:.0f}m ({constraints.max_range_km}km)"
            )

    for i, wp in enumerate(plan.waypoints):
        if wp.alt_m is not None:
            if wp.alt_m < constraints.min_altitude_m:
                raise ValueError(f"Waypoint {i} altitude {wp.alt_m}m below minimum {constraints.min_altitude_m}m")
            if wp.alt_m > constraints.max_altitude_m:
                raise ValueError(f"Waypoint {i} altitude {wp.alt_m}m above maximum {constraints.max_altitude_m}m")


def validate_waypoints_near_home(home: GeoPoint, waypoints: list[GeoPoint], max_distance_m: float = 500.0) -> None:
    max_dist = 0.0
    for i, wp in enumerate(waypoints):
        dist = haversine_distance_m(home, wp)
        if dist > max_dist:
            max_dist = dist
        if dist > max_distance_m:
            raise ValueError(
                f"Waypoint {i} is {dist:.0f}m from home — exceeds {max_distance_m:.0f}m limit. "
                f"Use --local-sitl-mission to generate a mission near the SITL home position."
            )
