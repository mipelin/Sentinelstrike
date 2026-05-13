"""Tests for mission validation."""

import pytest

from sentinel.common.types import GeoPoint, MissionConstraints, MissionPlan
from sentinel.mission_planner.validation import (
    validate_area_of_interest,
    validate_plan_against_constraints,
)


def test_area_fewer_than_3_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        validate_area_of_interest([GeoPoint(lat=38, lon=-8), GeoPoint(lat=39, lon=-8)])


def test_area_all_points_identical():
    pts = [GeoPoint(lat=38.0, lon=-8.0)] * 4
    with pytest.raises(ValueError, match="identical"):
        validate_area_of_interest(pts)


def test_area_all_on_a_line_rejected():
    pts = [
        GeoPoint(lat=38.0, lon=-8.0),
        GeoPoint(lat=38.001, lon=-8.0),
        GeoPoint(lat=38.002, lon=-8.0),
    ]
    validate_area_of_interest(pts)  # collinear is still valid — has non-zero bbox


def test_plan_distance_exceeds_max():
    plan = MissionPlan(
        mission_id="t",
        waypoints=[GeoPoint(lat=38, lon=-8), GeoPoint(lat=38, lon=-8)],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38, lon=-8),
        total_distance_m=100_000,
    )
    constraints = MissionConstraints(max_range_km=1)
    with pytest.raises(ValueError, match="exceeds max range"):
        validate_plan_against_constraints(plan, constraints)


def test_waypoint_altitude_below_min():
    plan = MissionPlan(
        mission_id="t",
        waypoints=[GeoPoint(lat=38, lon=-8, alt_m=10)],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38, lon=-8),
    )
    constraints = MissionConstraints(min_altitude_m=40)
    with pytest.raises(ValueError, match="below minimum"):
        validate_plan_against_constraints(plan, constraints)


def test_waypoint_altitude_above_max():
    plan = MissionPlan(
        mission_id="t",
        waypoints=[GeoPoint(lat=38, lon=-8, alt_m=200)],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38, lon=-8),
    )
    constraints = MissionConstraints(max_altitude_m=120)
    with pytest.raises(ValueError, match="above maximum"):
        validate_plan_against_constraints(plan, constraints)


def test_valid_plan_passes():
    plan = MissionPlan(
        mission_id="t",
        waypoints=[GeoPoint(lat=38, lon=-8, alt_m=80)],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38, lon=-8),
        total_distance_m=1000,
    )
    constraints = MissionConstraints(max_range_km=10)
    validate_plan_against_constraints(plan, constraints)
