"""Tests for TelemetryCache, FlightBridgeWorker, and MAVSDK isolation."""

from __future__ import annotations

import threading
import time

import pytest

from sentinel.common.types import GeoPoint
from sentinel.flight.telemetry_cache import TelemetryCache, TelemetrySnapshot
from sentinel.runtime.contracts import DroneState, FlightStatus, VelocityCommand
from sentinel.runtime.workers.flight_bridge_worker import FlightBridgeWorker


# ---------------------------------------------------------------------------
# TelemetryCache
# ---------------------------------------------------------------------------


class TestTelemetryCache:
    def test_initial_state(self):
        cache = TelemetryCache()
        pos, ts = cache.get_position()
        assert pos is None
        assert ts == 0.0
        assert not cache.is_connected()

    def test_update_position(self):
        cache = TelemetryCache()
        seq = cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        assert seq == 1
        pos, ts = cache.get_position()
        assert pos is not None
        assert pos.lat == 38.0
        assert pos.lon == -8.0
        assert pos.alt_m == 50.0
        assert ts > 0

    def test_update_heading(self):
        cache = TelemetryCache()
        cache.update_heading(180.0)
        hdg, ts = cache.get_heading()
        assert hdg == 180.0

    def test_update_armed(self):
        cache = TelemetryCache()
        cache.update_armed(True)
        armed, _ = cache.get_armed()
        assert armed is True

    def test_update_mode(self):
        cache = TelemetryCache()
        cache.update_mode("OFFBOARD")
        mode, _ = cache.get_mode()
        assert mode == "OFFBOARD"

    def test_update_battery(self):
        cache = TelemetryCache()
        cache.update_battery(85.5)
        bat, _ = cache.get_battery()
        assert bat == 85.5

    def test_update_connected(self):
        cache = TelemetryCache()
        cache.update_connected(True)
        assert cache.is_connected()

    def test_overwrite_semantics(self):
        cache = TelemetryCache()
        cache.update_position(lat=1.0, lon=2.0, alt_m=10.0)
        cache.update_position(lat=3.0, lon=4.0, alt_m=20.0)
        pos, _ = cache.get_position()
        assert pos.lat == 3.0  # latest only

    def test_age_freshness(self):
        cache = TelemetryCache()
        assert cache.age_s("position") == float("inf")
        assert not cache.is_fresh("position", 5.0)

        cache.update_position(lat=0.0, lon=0.0, alt_m=0.0)
        assert cache.age_s("position") < 1.0
        assert cache.is_fresh("position", 5.0)

        time.sleep(0.05)
        assert cache.age_s("position") >= 0.04

    def test_snapshot(self):
        cache = TelemetryCache()
        cache.update_connected(True)
        cache.update_armed(True)
        cache.update_mode("OFFBOARD")
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        cache.update_heading(90.0)
        cache.update_battery(75.0)

        snap = cache.snapshot()
        assert isinstance(snap, TelemetrySnapshot)
        assert snap.connected
        assert snap.armed
        assert snap.mode == "OFFBOARD"
        assert snap.position.lat == 38.0
        assert snap.heading_deg == 90.0
        assert snap.battery_pct == 75.0
        assert snap.position_age_s < 1.0

    def test_snapshot_staleness(self):
        cache = TelemetryCache()
        snap = cache.snapshot()
        assert snap.stale(5.0)  # no position yet

        cache.update_position(lat=0.0, lon=0.0, alt_m=0.0)
        snap = cache.snapshot()
        assert not snap.stale(5.0)
        assert snap.stale(0.0)  # threshold 0 always stale (age > 0)

    def test_thread_safety(self):
        cache = TelemetryCache()
        errors = []

        def writer(offset: int) -> None:
            try:
                for i in range(200):
                    lat = (offset + i) % 180 - 90  # keep in [-90, 90]
                    lon = (offset + i) % 360 - 180  # keep in [-180, 180]
                    cache.update_position(
                        lat=float(lat), lon=float(lon), alt_m=float(i),
                    )
                    cache.update_heading(float((offset + i) % 360))
                    cache.update_armed(i % 2 == 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t * 1000,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        pos, _ = cache.get_position()
        assert pos is not None

    def test_independent_field_ages(self):
        cache = TelemetryCache()
        cache.update_position(lat=1.0, lon=1.0, alt_m=1.0)
        time.sleep(0.05)
        cache.update_heading(90.0)

        assert cache.age_s("position") > 0.04
        assert cache.age_s("heading_deg") < 0.04


# ---------------------------------------------------------------------------
# FlightBridgeWorker (mock mode)
# ---------------------------------------------------------------------------


class TestFlightBridgeWorkerMock:
    def test_mock_connect(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        assert fbw.connected

    def test_mock_engage_offboard(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        assert fbw.engage_offboard()
        assert fbw.offboard_active

    def test_mock_disengage_offboard(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        fbw.engage_offboard()
        assert fbw.disengage_offboard()
        assert not fbw.offboard_active

    def test_telemetry_slot_published_on_tick(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=100.0)
        fbw.telemetry_cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        fbw.telemetry_cache.update_heading(180.0)
        fbw.tick()

        state_val, seq, _ = fbw.telemetry_slot.read()
        assert state_val is not None
        assert isinstance(state_val, DroneState)
        assert seq == 1
        assert state_val.position is not None
        assert state_val.position["lat"] == 38.0

    def test_flight_status_published_on_tick(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=100.0)
        fbw.telemetry_cache.update_armed(True)
        fbw.telemetry_cache.update_mode("OFFBOARD")
        fbw.tick()

        status_val, _, _ = fbw.flight_status_slot.read()
        assert isinstance(status_val, FlightStatus)
        assert status_val.state == "OFFBOARD"

    def test_stale_command_sends_hold(self):
        fbw = FlightBridgeWorker(
            backend="mock", mock_connected=True, hz=100.0,
            command_stale_s=0.1,
        )
        fbw.engage_offboard()

        # Write a command that's already old
        fbw.command_slot.write(VelocityCommand(vx=1.0, vy=0.0, vz=0.0, yawspeed=0.0))
        time.sleep(0.15)  # age the command past stale threshold

        fbw.tick()
        assert fbw._dropped_stale_commands == 1

    def test_fresh_command_accepted(self):
        fbw = FlightBridgeWorker(
            backend="mock", mock_connected=True, hz=100.0,
            command_stale_s=5.0,
        )
        fbw.engage_offboard()

        fbw.command_slot.write(VelocityCommand(vx=0.5, vy=-0.3, vz=0.0, yawspeed=0.1))
        fbw.tick()
        assert fbw._dropped_stale_commands == 0

    def test_command_overwrite_semantics(self):
        """Only latest command matters — no queue."""
        fbw = FlightBridgeWorker(
            backend="mock", mock_connected=True, hz=100.0,
            command_stale_s=5.0,
        )
        fbw.engage_offboard()

        # Write multiple commands rapidly
        for i in range(10):
            fbw.command_slot.write(
                VelocityCommand(vx=float(i), vy=0.0, vz=0.0, yawspeed=0.0),
            )

        fbw.tick()
        # Only the latest command was seen
        cmd_val, cmd_seq, _ = fbw.command_slot.read()
        assert cmd_seq == 10  # 10 writes, but LatestSlot only keeps latest
        assert cmd_val.vx == 9.0

    def test_start_stop_lifecycle(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)
        assert not fbw.running
        fbw.start()
        assert fbw.running
        time.sleep(0.1)
        fbw.stop()
        assert not fbw.running

    def test_worker_survives_exceptions(self):
        """Worker continues even if tick throws internally."""
        class FailingBridge(FlightBridgeWorker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._tick_count = 0

            def _send_velocity(self, vx, vy, vz, yawspeed):
                self._tick_count += 1
                if self._tick_count <= 2:
                    raise RuntimeError("MAVSDK send failure")

        fbw = FailingBridge(backend="mock", mock_connected=True, hz=100.0)
        fbw.engage_offboard()
        fbw.command_slot.write(VelocityCommand(vx=1.0))
        fbw.start()
        time.sleep(0.3)
        fbw.stop()

        assert fbw._metrics.exceptions >= 2
        assert fbw._metrics.total_ticks >= 3  # continued after exceptions

    def test_health_report(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)
        fbw.telemetry_cache.update_position(lat=0.0, lon=0.0, alt_m=0.0)
        fbw.start()
        time.sleep(0.1)
        health = fbw.health_report()
        assert health["connected"]
        assert "worker_metrics" in health
        assert "dropped_stale_commands" in health
        fbw.stop()

    def test_offboard_loop_cadence(self):
        """Offboard loop runs at target Hz."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)
        fbw.engage_offboard()
        fbw.start()
        time.sleep(0.5)
        fbw.stop()

        # Should have ~25 ticks in 0.5s at 50Hz (allowing for startup)
        assert fbw._metrics.total_ticks >= 15

    def test_telemetry_freeze_isolation(self):
        """Stale telemetry doesn't crash the vision loop."""
        cache = TelemetryCache()
        cache.update_position(lat=1.0, lon=1.0, alt_m=1.0)
        cache.update_heading(0.0)

        # Simulate telemetry freeze — no updates for a while
        time.sleep(0.05)

        snap = cache.snapshot()
        assert snap.stale(0.01)  # stale by now
        # But reading still works — no exception
        assert snap.position is not None

    def test_no_command_no_crash(self):
        """Worker handles no-command gracefully."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=100.0)
        fbw.engage_offboard()
        fbw.tick()  # no command written
        assert fbw._dropped_stale_commands == 0
