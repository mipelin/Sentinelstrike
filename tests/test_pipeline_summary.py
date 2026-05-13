"""Tests for pipeline summary."""

from pathlib import Path

from sentinel.common.types import GeoPoint, MissionPlan
from sentinel.pipeline.summary import build_pipeline_summary


def _plan():
    return MissionPlan(
        mission_id="m1",
        waypoints=[GeoPoint(lat=38.0, lon=-8.0, alt_m=80), GeoPoint(lat=38.01, lon=-8.0, alt_m=80)],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
        total_distance_m=1113.0,
        estimated_duration_s=111.3,
    )


def test_summary_includes_mission_id():
    s = build_pipeline_summary(mission_id="m1", run_dir=Path("/tmp/r"), mission_plan=_plan())
    assert s["mission_id"] == "m1"


def test_summary_planner_waypoint_count():
    s = build_pipeline_summary(mission_id="m1", run_dir=Path("/tmp/r"), mission_plan=_plan())
    assert s["planner"]["waypoint_count"] == 2
    assert s["planner"]["total_distance_m"] == 1113.0


def test_summary_missing_metrics():
    s = build_pipeline_summary(
        mission_id="m1",
        run_dir=Path("/tmp/r"),
        mission_plan=_plan(),
        perception_metrics=None,
        tracker_metrics=None,
    )
    assert s["perception"]["frame_count"] == 0
    assert s["tracking"]["total_tracks"] == 0


def test_summary_artifact_keys():
    s = build_pipeline_summary(mission_id="m1", run_dir=Path("/tmp/r"), mission_plan=_plan())
    assert "events" in s["artifacts"]
    assert "mission_plan" in s["artifacts"]
    assert "report" in s["artifacts"]
