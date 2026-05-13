"""Mission planner v1 — generates search patterns from mission requests."""

from __future__ import annotations

from loguru import logger

from sentinel.common.geo import bounding_box, centroid, offset_point
from sentinel.common.types import GeoPoint, MissionConstraints, MissionPlan, MissionRequest

from .metrics import plan_metrics
from .patterns import generate_lawnmower_pattern, generate_perimeter_pattern
from .validation import validate_area_of_interest, validate_plan_against_constraints


class MissionPlanner:
    def create_plan(self, request: MissionRequest) -> MissionPlan:
        validate_area_of_interest(request.area_of_interest)

        constraints = request.constraints
        altitude_m = constraints.default_altitude_m
        bbox = bounding_box(request.area_of_interest)

        if request.search_pattern == "lawnmower":
            pattern_points = generate_lawnmower_pattern(
                area_of_interest=request.area_of_interest,
                spacing_m=constraints.search_spacing_m,
                altitude_m=altitude_m,
                max_waypoints=constraints.max_waypoints,
            )
        elif request.search_pattern == "perimeter":
            pattern_points = generate_perimeter_pattern(
                area_of_interest=request.area_of_interest,
                altitude_m=altitude_m,
            )
        else:
            raise ValueError(f"Unsupported search pattern: {request.search_pattern}")

        launch_alt = altitude_m
        launch_wp = GeoPoint(
            lat=request.launch_point.lat,
            lon=request.launch_point.lon,
            alt_m=launch_alt,
        )
        return_wp = GeoPoint(
            lat=request.launch_point.lat,
            lon=request.launch_point.lon,
            alt_m=launch_alt,
        )

        waypoints = [launch_wp, *pattern_points, return_wp]

        center = centroid(request.area_of_interest)
        center_wp = GeoPoint(lat=center.lat, lon=center.lon, alt_m=altitude_m)

        metrics = plan_metrics(waypoints, constraints.default_speed_mps)

        plan = MissionPlan(
            mission_id=request.mission_id,
            waypoints=waypoints,
            search_pattern=request.search_pattern,
            abort_points=[launch_wp, center_wp],
            return_home_point=return_wp,
            operator_checkpoints=[center_wp],
            total_distance_m=metrics["total_distance_m"],
            estimated_duration_s=metrics["estimated_duration_s"],
            metadata={
                "pattern": request.search_pattern,
                "planner": "mission_planner_v1",
                "search_spacing_m": constraints.search_spacing_m,
                "default_altitude_m": constraints.default_altitude_m,
                "default_speed_mps": constraints.default_speed_mps,
                "bbox": bbox,
            },
        )

        validate_plan_against_constraints(plan, constraints)

        logger.info(
            "Plan created: {} waypoints, {:.0f}m, {:.0f}s",
            metrics["waypoint_count"],
            metrics["total_distance_m"],
            metrics["estimated_duration_s"],
        )

        return plan


def build_sitl_local_request(
    home: GeoPoint,
    size_m: float = 80.0,
    altitude_m: float = 30.0,
    spacing_m: float = 30.0,
    speed_mps: float = 5.0,
    max_range_km: float = 2.0,
) -> MissionRequest:
    """Build a MissionRequest for a small area around the given home position.

    Used for PX4 SITL tests where the home position is the SITL default
    (e.g. Zurich 47.3979704, 8.5461632) and not the Portugal coordinates
    from the demo mission JSON.
    """
    half = size_m / 2
    corners = [
        offset_point(home, -half, -half, altitude_m),
        offset_point(home, -half, half, altitude_m),
        offset_point(home, half, half, altitude_m),
        offset_point(home, half, -half, altitude_m),
    ]
    return MissionRequest(
        mission_id="sitl_local",
        launch_point=GeoPoint(lat=home.lat, lon=home.lon, alt_m=0),
        area_of_interest=corners,
        search_pattern="lawnmower",
        constraints=MissionConstraints(
            max_altitude_m=120,
            min_altitude_m=20,
            max_range_km=max_range_km,
            search_spacing_m=spacing_m,
            default_altitude_m=altitude_m,
            default_speed_mps=speed_mps,
            max_waypoints=50,
        ),
    )
