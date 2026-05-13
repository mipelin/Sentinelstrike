"""Tests for MavsdkBackend import behavior — does not require PX4 or mavsdk."""

import inspect

import pytest


def test_import_does_not_break_without_mavsdk():
    """Importing the module must not fail even if mavsdk is not installed."""
    from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

    assert MavsdkBackend is not None


def test_instantiation_defers_mavsdk_import():
    """Creating an instance should not fail — mavsdk is only imported on connect()."""
    from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

    backend = MavsdkBackend(vehicle_id="test")
    assert backend._connection_url == "udpin://0.0.0.0:14540"
    backend.close()


def test_connect_fails_clearly_without_mavsdk():
    """If mavsdk is not installed, connect() must return a failed CommandResult with install instructions."""
    import sentinel.mavlink_bridge.mavsdk_backend as mod

    # Only test if mavsdk is NOT installed
    try:
        import mavsdk  # noqa: F401

        pytest.skip("mavsdk is installed — skipping no-mavsdk test")
    except ImportError:
        pass

    backend = mod.MavsdkBackend(vehicle_id="test")
    result = backend.connect()
    assert result.success is False
    assert "pip install -e" in result.message
    backend.close()


def test_no_asyncio_run_in_backend():
    """mavsdk_backend.py must NOT call asyncio.run() as a function — it breaks MAVSDK gRPC channels."""
    import ast

    import sentinel.mavlink_bridge.mavsdk_backend as mod

    source = inspect.getsource(mod)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
            ):
                raise AssertionError(
                    "mavsdk_backend.py must not call asyncio.run() — use AsyncLoopThread.run() instead"
                )


def test_mission_item_values_helper():
    """_mission_item_values_from_waypoint returns all required fields."""
    from sentinel.mavlink_bridge.mavsdk_backend import _mission_item_values_from_waypoint

    vals = _mission_item_values_from_waypoint(
        wp_lat=38.0,
        wp_lon=-8.0,
        wp_alt_m=60.0,
        default_alt=50.0,
        speed=10.0,
    )

    assert vals["latitude_deg"] == 38.0
    assert vals["longitude_deg"] == -8.0
    assert vals["relative_altitude_m"] == 60.0
    assert vals["speed_m_s"] == 10.0
    assert vals["is_fly_through"] is True
    assert vals["camera_photo_interval_s"] != vals["camera_photo_interval_s"]  # NaN check
    assert vals["acceptance_radius_m"] == 5.0
    assert vals["yaw_deg"] != vals["yaw_deg"]  # NaN
    assert vals["camera_photo_distance_m"] != vals["camera_photo_distance_m"]  # NaN
    assert vals["vehicle_action"] == "NONE"
    assert vals["camera_action"] == "NONE"


def test_mission_item_values_uses_default_alt():
    """When wp_alt_m is None, the default altitude is used."""
    from sentinel.mavlink_bridge.mavsdk_backend import _mission_item_values_from_waypoint

    vals = _mission_item_values_from_waypoint(
        wp_lat=38.0,
        wp_lon=-8.0,
        wp_alt_m=None,
        default_alt=50.0,
        speed=10.0,
    )
    assert vals["relative_altitude_m"] == 50.0
