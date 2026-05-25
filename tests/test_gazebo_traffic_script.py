"""Tests for run_gazebo_traffic waypoint math and interpolation."""

from __future__ import annotations

import importlib
import math
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_gz(monkeypatch):
    """Patch gz imports so the module loads without Gazebo."""
    mock_gz = types.ModuleType("gz")
    mock_transport = types.ModuleType("gz.transport13")
    mock_msgs = types.ModuleType("gz.msgs10")
    mock_pose = types.ModuleType("gz.msgs10.pose_pb2")
    mock_bool = types.ModuleType("gz.msgs10.boolean_pb2")

    class _MockPose:
        def __init__(self):
            self.name = ""
            self.position = MagicMock()
            self.orientation = MagicMock()

    mock_pose.Pose = _MockPose
    mock_bool.Boolean = MagicMock

    mock_gz.transport13 = mock_transport
    mock_gz.msgs10 = mock_msgs
    mock_msgs.pose_pb2 = mock_pose
    mock_msgs.boolean_pb2 = mock_bool

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.pose_pb2", mock_pose)
    monkeypatch.setitem(sys.modules, "gz.msgs10.boolean_pb2", mock_bool)

    for mod in list(sys.modules):
        if "run_gazebo_traffic" in mod:
            del sys.modules[mod]


def _get_module():
    return importlib.import_module("apps.tools.run_gazebo_traffic")


class TestWaypointInterpolation:
    def test_single_segment_forward(self):
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=100, y=0, z=0, yaw=0, speed=10),
            ],
        )
        # 100m at 10 m/s = 10s total
        result = mod._interpolate_route(route, 0)
        assert result is not None
        assert abs(result[0] - 0.0) < 0.01  # x at t=0

        result = mod._interpolate_route(route, 5)
        assert abs(result[0] - 50.0) < 0.01  # x at t=5

        result = mod._interpolate_route(route, 10)
        # At t=10 (== total_time), loop wraps to t=0
        assert abs(result[0] - 0.0) < 0.01  # wraps to start

        # Just before the end, should be near 100
        result = mod._interpolate_route(route, 9.99)
        assert abs(result[0] - 99.9) < 0.5

    def test_loop_wraps(self):
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=100, y=0, z=0, yaw=0, speed=10),
            ],
        )
        # t=15 should wrap: 15 % 10 = 5, so x=50
        result = mod._interpolate_route(route, 15)
        assert abs(result[0] - 50.0) < 0.01

    def test_no_loop_clamps(self):
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            loop=False,
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=100, y=0, z=0, yaw=0, speed=10),
            ],
        )
        result = mod._interpolate_route(route, 100)
        assert abs(result[0] - 100.0) < 0.01  # clamped at end

    def test_multi_segment(self):
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=50, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=50, y=50, z=0, yaw=0, speed=10),
            ],
        )
        # Segment 1: 50m at 10 m/s = 5s
        # Segment 2: 50m at 10 m/s = 5s
        # Total: 10s

        result = mod._interpolate_route(route, 2.5)
        assert abs(result[0] - 25.0) < 0.01
        assert abs(result[1] - 0.0) < 0.01

        result = mod._interpolate_route(route, 7.5)
        assert abs(result[0] - 50.0) < 0.01
        assert abs(result[1] - 25.0) < 0.01

    def test_yaw_interpolation(self):
        """Yaw is computed from velocity direction (atan2), not linear interp."""
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=10, y=0, z=0, yaw=1.57, speed=10),
            ],
        )
        # Movement is purely +X, so atan2(0,10)=0 regardless of waypoint yaw
        result = mod._interpolate_route(route, 0.5)
        assert abs(result[3] - 0.0) < 0.01  # heading = +X direction

    def test_yaw_from_velocity_diagonal(self):
        """Diagonal movement gives atan2 heading."""
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=10, y=10, z=0, yaw=0, speed=10),
            ],
        )
        result = mod._interpolate_route(route, 0.5)
        assert abs(result[3] - math.pi / 4) < 0.01  # atan2(10,10) = pi/4

    def test_yaw_from_velocity_north(self):
        """Pure northward movement gives pi/2 heading."""
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10),
                mod.Waypoint(x=0, y=10, z=0, yaw=0, speed=10),
            ],
        )
        result = mod._interpolate_route(route, 0.5)
        assert abs(result[3] - math.pi / 2) < 0.01  # atan2(10,0) = pi/2

    def test_single_waypoint_returns_none(self):
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=10)],
        )
        assert mod._interpolate_route(route, 0) is None

    def test_zero_speed_uses_fallback(self):
        mod = _get_module()
        route = mod.Route(
            model_name="test",
            waypoints=[
                mod.Waypoint(x=0, y=0, z=0, yaw=0, speed=0),
                mod.Waypoint(x=100, y=0, z=0, yaw=0, speed=0),
            ],
        )
        # speed=0 falls back to 10 m/s
        result = mod._interpolate_route(route, 5)
        assert abs(result[0] - 50.0) < 0.01


class TestDefaultRoutes:
    def test_four_routes_defined_heavy(self):
        mod = _get_module()
        routes = mod._default_routes("heavy")
        assert len(routes) == 4

    def test_medium_returns_two(self):
        mod = _get_module()
        routes = mod._default_routes("medium")
        assert len(routes) == 2

    def test_light_returns_one(self):
        mod = _get_module()
        routes = mod._default_routes("light")
        assert len(routes) == 1

    def test_route_model_names_heavy(self):
        mod = _get_module()
        routes = mod._default_routes("heavy")
        names = [r.model_name for r in routes]
        assert "vehicle_moving_1" in names
        assert "vehicle_moving_2" in names
        assert "vehicle_moving_3" in names
        assert "vehicle_moving_4" in names

    def test_all_routes_have_minimum_waypoints(self):
        mod = _get_module()
        routes = mod._default_routes("heavy")
        for r in routes:
            assert len(r.waypoints) >= 2, f"{r.model_name} needs at least 2 waypoints"

    def test_routes_loop_by_default(self):
        mod = _get_module()
        routes = mod._default_routes("heavy")
        for r in routes:
            assert r.loop is True

    def test_default_routes_z_zero(self):
        """All waypoints must have Z=0 (Fuel models sit on ground plane)."""
        mod = _get_module()
        routes = mod._default_routes("heavy")
        for r in routes:
            for wp in r.waypoints:
                assert wp.z == 0.0, f"{r.model_name} waypoint has z={wp.z}"

    def test_default_routes_have_yaw_offset(self):
        """Each route has a yaw_offset for Fuel model compensation.

        Corrected offsets derived from actual model.sdf visual rotations:
          Pickup:  model SDF visual rot -π/2 reorients mesh +Y→+X, internal_yaw=0
          Hatchback: model SDF visual rot +π/2 reorients mesh -Y→+X, internal_yaw=0
          Prius: collision boxes show front at -Y, internal_yaw=-π/2, offset=+π/2
          TruckBox: mesh forward along +X, internal_yaw=0
        """
        mod = _get_module()
        routes = mod._default_routes("heavy")
        offsets = {r.model_name: r.yaw_offset for r in routes}
        assert offsets["vehicle_moving_1"] == 0.0            # Pickup — internal_yaw=0
        assert offsets["vehicle_moving_2"] == 0.0            # Hatchback — internal_yaw=0
        assert offsets["vehicle_moving_3"] == math.pi / 2    # Prius — internal_yaw=-π/2
        assert offsets["vehicle_moving_4"] == 0.0            # TruckBox — internal_yaw=0


class TestMakePoseMsg:
    def test_yaw_zero_quaternion(self):
        mod = _get_module()
        msg = mod._make_pose_msg("test", 1, 2, 3, 0)
        assert msg.name == "test"
        assert abs(msg.orientation.w - 1.0) < 0.01
        assert abs(msg.orientation.z - 0.0) < 0.01

    def test_yaw_pi_quaternion(self):
        mod = _get_module()
        msg = mod._make_pose_msg("test", 0, 0, 0, math.pi)
        assert abs(msg.orientation.z - math.sin(math.pi / 2)) < 0.01
        assert abs(msg.orientation.w - math.cos(math.pi / 2)) < 0.01
