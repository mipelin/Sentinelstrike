"""Tests for the mission planner v1."""

import pytest

from sentinel.common.geo import haversine_distance_m
from sentinel.common.types import GeoPoint, MissionConstraints, MissionRequest
from sentinel.mission_planner.planner import MissionPlanner, build_sitl_local_request
from sentinel.mission_planner.validation import validate_waypoints_near_home


def _area():
    return [
        GeoPoint(lat=38.000, lon=-8.000, alt_m=80),
        GeoPoint(lat=38.000, lon=-7.998, alt_m=80),
        GeoPoint(lat=38.002, lon=-7.998, alt_m=80),
        GeoPoint(lat=38.002, lon=-8.000, alt_m=80),
    ]


def _request(**overrides):
    defaults = dict(
        mission_id="t1",
        launch_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=0),
        area_of_interest=_area(),
    )
    defaults.update(overrides)
    return MissionRequest(**defaults)


def test_create_plan_returns_mission_plan():
    plan = MissionPlanner().create_plan(_request())
    assert plan.mission_id == "t1"
    assert len(plan.waypoints) > 0


def test_waypoints_start_and_end_at_launch():
    req = _request()
    plan = MissionPlanner().create_plan(req)
    first = plan.waypoints[0]
    last = plan.waypoints[-1]
    assert first.lat == req.launch_point.lat
    assert first.lon == req.launch_point.lon
    assert last.lat == req.launch_point.lat
    assert last.lon == req.launch_point.lon


def test_return_home_equals_launch():
    req = _request()
    plan = MissionPlanner().create_plan(req)
    assert plan.return_home_point.lat == req.launch_point.lat
    assert plan.return_home_point.lon == req.launch_point.lon


def test_operator_checkpoints_has_centroid():
    plan = MissionPlanner().create_plan(_request())
    assert len(plan.operator_checkpoints) >= 1
    cp = plan.operator_checkpoints[0]
    assert 37.9 < cp.lat < 38.1
    assert -8.1 < cp.lon < -7.9


def test_total_distance_positive():
    plan = MissionPlanner().create_plan(_request())
    assert plan.total_distance_m is not None
    assert plan.total_distance_m > 0


def test_estimated_duration_positive():
    plan = MissionPlanner().create_plan(_request())
    assert plan.estimated_duration_s is not None
    assert plan.estimated_duration_s > 0


def test_metadata_includes_pattern_and_bbox():
    plan = MissionPlanner().create_plan(_request())
    assert plan.metadata["pattern"] == "lawnmower"
    assert "bbox" in plan.metadata
    assert "min_lat" in plan.metadata["bbox"]


def test_lawnmower_generates_many_waypoints():
    plan = MissionPlanner().create_plan(_request())
    assert len(plan.waypoints) > 6


def test_unsupported_pattern_raises():
    with pytest.raises(ValueError, match="Unsupported search pattern"):
        MissionPlanner().create_plan(_request(search_pattern="spiral"))


def test_max_range_exceeded_raises():
    constraints = MissionConstraints(max_range_km=0.001)
    with pytest.raises(ValueError, match="exceeds max range"):
        MissionPlanner().create_plan(_request(constraints=constraints))


def test_planner_version():
    plan = MissionPlanner().create_plan(_request())
    assert plan.planner_version == "mission_planner_v1"


# ---------------------------------------------------------------------------
# SITL local mission tests
# ---------------------------------------------------------------------------


def test_sitl_local_request_generates_plan():
    """build_sitl_local_request produces a valid plan around the given home."""
    home = GeoPoint(lat=47.3979704, lon=8.5461632, alt_m=0)
    request = build_sitl_local_request(home)
    plan = MissionPlanner().create_plan(request)
    assert plan.mission_id == "sitl_local"
    assert len(plan.waypoints) > 0


def test_sitl_local_waypoints_near_home():
    """All waypoints in the local SITL mission are within 100m of home."""
    home = GeoPoint(lat=47.3979704, lon=8.5461632, alt_m=0)
    request = build_sitl_local_request(home, size_m=80)
    plan = MissionPlanner().create_plan(request)
    for wp in plan.waypoints:
        dist = haversine_distance_m(home, wp)
        assert dist < 100, f"Waypoint at {wp.lat},{wp.lon} is {dist:.1f}m from home — expected <100m"


def test_sitl_local_first_waypoint_near_home():
    """First waypoint is at the launch point (home)."""
    home = GeoPoint(lat=47.3979704, lon=8.5461632, alt_m=0)
    request = build_sitl_local_request(home)
    plan = MissionPlanner().create_plan(request)
    first = plan.waypoints[0]
    dist = haversine_distance_m(home, first)
    assert dist < 1.0, f"First waypoint {dist:.1f}m from home — expected <1m"


def test_validate_waypoints_near_home_passes():
    """Validation passes when waypoints are close to home."""
    home = GeoPoint(lat=47.3979704, lon=8.5461632, alt_m=0)
    request = build_sitl_local_request(home, size_m=80)
    plan = MissionPlanner().create_plan(request)
    validate_waypoints_near_home(home, plan.waypoints, max_distance_m=500)


def test_validate_waypoints_near_home_aborts_distant_mission():
    """Validation aborts when mission waypoints are far from home (e.g. Portugal)."""
    home = GeoPoint(lat=47.3979704, lon=8.5461632, alt_m=0)
    distant_request = _request()
    plan = MissionPlanner().create_plan(distant_request)
    with pytest.raises(ValueError, match="exceeds"):
        validate_waypoints_near_home(home, plan.waypoints, max_distance_m=500)


def test_sitl_local_request_custom_params():
    """Custom parameters are respected in the generated request."""
    home = GeoPoint(lat=47.3979704, lon=8.5461632, alt_m=0)
    request = build_sitl_local_request(
        home,
        size_m=100,
        altitude_m=50,
        spacing_m=20,
        speed_mps=3,
        max_range_km=1,
    )
    assert request.constraints.default_altitude_m == 50
    assert request.constraints.default_speed_mps == 3
    assert request.constraints.search_spacing_m == 20
    assert request.constraints.max_range_km == 1
