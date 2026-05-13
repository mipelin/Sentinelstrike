"""Tests for dashboard map layer builder."""

from sentinel.dashboard.map_layers import build_map_layers


def test_empty_artifacts():
    layers = build_map_layers({})
    assert layers.waypoints == []
    assert layers.observations == []
    assert layers.uav_positions == []
    assert layers.safety_events == []


def test_waypoints_from_mission_plan():
    artifacts = {
        "mission_plan": {
            "waypoints": [
                {"lat": 38.0, "lon": -8.0},
                {"lat": 38.01, "lon": -8.01},
            ],
        },
    }
    layers = build_map_layers(artifacts)
    assert len(layers.waypoints) == 2
    assert layers.waypoints[0].lat == 38.0
    assert layers.waypoints[0].label == "WP 1"
    assert layers.waypoints[1].label == "WP 2"


def test_observations_from_geo():
    artifacts = {
        "geo_observations": [
            {
                "observation_id": "obs_1",
                "estimated_location": {"lat": 38.0, "lon": -8.0},
                "track_id": "trk_1",
                "class_name": "person",
                "confidence": 0.9,
            },
        ],
    }
    layers = build_map_layers(artifacts)
    assert len(layers.observations) == 1
    assert layers.observations[0].confirmed is False
    assert layers.observations[0].class_name == "person"


def test_confirmed_from_operator_decisions():
    artifacts = {
        "geo_observations": [
            {
                "observation_id": "obs_1",
                "estimated_location": {"lat": 38.0, "lon": -8.0},
                "class_name": "person",
                "confidence": 0.9,
            },
        ],
        "operator_decisions": [
            {"action": "CONFIRM_OBSERVATION", "observation_id": "obs_1"},
            {"action": "REJECT_OBSERVATION", "observation_id": "obs_2"},
        ],
    }
    layers = build_map_layers(artifacts)
    assert len(layers.observations) == 1
    assert layers.observations[0].confirmed is True


def test_safety_events():
    artifacts = {
        "safety_actions": [
            {"action": "LAND", "trigger": "OPERATOR_ABORT"},
            {"action": "RETURN_HOME", "trigger": "LINK_LOSS"},
        ],
    }
    layers = build_map_layers(artifacts)
    assert len(layers.safety_events) == 2
    assert layers.safety_events[0].action == "LAND"


def test_fallback_uav_from_waypoints():
    artifacts = {
        "mission_plan": {
            "waypoints": [{"lat": 38.5, "lon": -8.5}],
        },
    }
    layers = build_map_layers(artifacts)
    assert len(layers.uav_positions) == 1
    assert layers.uav_positions[0].lat == 38.5


def test_uav_from_metadata():
    artifacts = {
        "metadata": {
            "vehicle_position": {"lat": 38.2, "lon": -8.2},
        },
    }
    layers = build_map_layers(artifacts)
    assert len(layers.uav_positions) == 1
    assert layers.uav_positions[0].lat == 38.2
