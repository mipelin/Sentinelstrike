"""Tests for offboard methods on MavsdkBackend, MockMavlinkBackend, and follow self-test."""

from __future__ import annotations

import inspect

import pytest

from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend


# ---------------------------------------------------------------------------
# MockMavlinkBackend offboard tests
# ---------------------------------------------------------------------------


class TestMockOffboard:
    def test_offboard_start_when_connected(self):
        b = MockMavlinkBackend()
        b.connect()
        r = b.offboard_start()
        assert r.success
        assert b.offboard_is_active()

    def test_offboard_start_when_not_connected(self):
        b = MockMavlinkBackend()
        r = b.offboard_start()
        assert not r.success

    def test_offboard_stop(self):
        b = MockMavlinkBackend()
        b.connect()
        b.offboard_start()
        r = b.offboard_stop()
        assert r.success
        assert not b.offboard_is_active()

    def test_offboard_stop_idempotent(self):
        b = MockMavlinkBackend()
        r = b.offboard_stop()
        assert r.success
        assert not b.offboard_is_active()

    def test_set_velocity_when_active(self):
        b = MockMavlinkBackend()
        b.connect()
        b.offboard_start()
        r = b.offboard_set_velocity_body(1.0, 0.5, -0.3, 0.1)
        assert r.success
        assert b._last_velocity == (1.0, 0.5, -0.3, 0.1)

    def test_set_velocity_when_not_active(self):
        b = MockMavlinkBackend()
        b.connect()
        r = b.offboard_set_velocity_body(1.0, 0.0, 0.0, 0.0)
        assert not r.success

    def test_offboard_is_active_false_by_default(self):
        b = MockMavlinkBackend()
        assert not b.offboard_is_active()

    def test_offboard_lifecycle(self):
        b = MockMavlinkBackend()
        b.connect()
        assert not b.offboard_is_active()
        b.offboard_start()
        assert b.offboard_is_active()
        b.offboard_set_velocity_body(0.5, 0.5, 0.0, 0.2)
        b.offboard_stop()
        assert not b.offboard_is_active()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_mock_backend_has_all_offboard_methods(self):
        from sentinel.mavlink_bridge.backends import MavlinkBackend

        b = MockMavlinkBackend()
        assert isinstance(b, MavlinkBackend)

    def test_mock_backend_offboard_method_signatures(self):
        b = MockMavlinkBackend()
        assert callable(b.offboard_start)
        assert callable(b.offboard_stop)
        assert callable(b.offboard_set_velocity_body)
        assert callable(b.offboard_is_active)


# ---------------------------------------------------------------------------
# MavsdkBackend offboard method existence (no PX4 needed)
# ---------------------------------------------------------------------------


class TestMavsdkBackendOffboardMethods:
    def test_has_offboard_start(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        assert hasattr(b, "offboard_start")
        b.close()

    def test_has_offboard_stop(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        assert hasattr(b, "offboard_stop")
        b.close()

    def test_has_offboard_set_velocity_body(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        assert hasattr(b, "offboard_set_velocity_body")
        b.close()

    def test_has_offboard_is_active(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        assert hasattr(b, "offboard_is_active")
        b.close()

    def test_offboard_is_active_false_initially(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        assert not b.offboard_is_active()
        b.close()

    def test_backend_class_file_path(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        path = inspect.getfile(MavsdkBackend)
        assert "mavsdk_backend" in path

    def test_offboard_start_fails_without_connect(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        r = b.offboard_start()
        assert not r.success
        b.close()


# ---------------------------------------------------------------------------
# Startup self-test pattern (same logic as run_gazebo_yolo_test.py)
# ---------------------------------------------------------------------------


class TestFollowSelfTest:
    _REQUIRED_OFFBOARD = ("offboard_start", "offboard_stop",
                          "offboard_set_velocity_body", "offboard_is_active")

    def test_mock_backend_passes_self_test(self):
        b = MockMavlinkBackend()
        missing = [m for m in self._REQUIRED_OFFBOARD if not hasattr(b, m)]
        assert missing == []

    def test_mavsdk_backend_passes_self_test(self):
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        b = MavsdkBackend(vehicle_id="test")
        missing = [m for m in self._REQUIRED_OFFBOARD if not hasattr(b, m)]
        assert missing == []
        b.close()

    def test_object_without_offboard_fails_self_test(self):
        class NoOffboard:
            pass

        b = NoOffboard()
        missing = [m for m in self._REQUIRED_OFFBOARD if not hasattr(b, m)]
        assert len(missing) == 4


# ---------------------------------------------------------------------------
# CommandType enum
# ---------------------------------------------------------------------------


class TestOffboardCommandTypes:
    def test_offboard_command_types_exist(self):
        from sentinel.mavlink_bridge.commands import CommandType

        assert CommandType.OFFBOARD_START == "OFFBOARD_START"
        assert CommandType.OFFBOARD_STOP == "OFFBOARD_STOP"
        assert CommandType.OFFBOARD_SET_VELOCITY == "OFFBOARD_SET_VELOCITY"
