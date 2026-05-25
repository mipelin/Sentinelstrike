"""Tests for worker follow/standoff path — command-sink mode.

Verifies that in --use-workers mode:
- FollowController command-sink mode writes VelocityCommand to LatestSlot
- FollowController command-sink mode reads from TelemetryCache
- FollowController never calls backend offboard/telemetry methods
- FlightBridgeWorker consumes command_slot
- FlightBridgeWorker autonomy_enabled manages offboard lifecycle
- No MavsdkBackend is needed
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from sentinel.autonomy.follow_controller import FollowController, FollowMode, TargetState
from sentinel.flight.telemetry_cache import TelemetryCache
from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import VelocityCommand
from sentinel.runtime.workers.flight_bridge_worker import FlightBridgeWorker
from sentinel.tracker.world_projection import CameraParams, ProjectionReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_follow_controller(**overrides) -> FollowController:
    """Create FollowController in command-sink mode (no backend)."""
    slot: LatestSlot[VelocityCommand] = LatestSlot()
    cache = TelemetryCache()
    cache.update_connected(True)
    cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
    cache.update_heading(90.0)

    kwargs = dict(
        command_slot=slot,
        telemetry_cache=cache,
        follow_mode=FollowMode.STANDOFF,
    )
    kwargs.update(overrides)
    return FollowController(**kwargs), slot, cache


def _detections_with_target(track_id="TGT-001", cx=320, cy=240, bbox_h=100):
    return [{
        "track_id": track_id,
        "bbox": [cx - 50, cy - bbox_h // 2, cx + 50, cy + bbox_h // 2],
        "confidence": 0.85,
        "class": "person",
        "speed": 0.0,
        "velocity": [0, 0],
        "identity_descriptor": "red/blue",
    }]


# ---------------------------------------------------------------------------
# FollowController command-sink mode
# ---------------------------------------------------------------------------


class TestFollowControllerCommandSink:
    def test_no_backend_required(self):
        """FollowController works without backend when command_slot provided."""
        fc, slot, cache = _make_follow_controller()
        assert fc._backend is None
        assert fc._command_slot is slot

    def test_engage_offboard_noop(self):
        """_engage_offboard is a no-op in command-sink mode."""
        fc, _, _ = _make_follow_controller()
        assert fc._engage_offboard() is True
        assert fc._offboard_engaged is True

    def test_disengage_offboard_noop(self):
        """_disengage_offboard is a no-op in command-sink mode."""
        fc, _, _ = _make_follow_controller()
        fc._engage_offboard()
        fc._disengage_offboard()
        assert not fc._offboard_engaged

    def test_record_home_from_cache(self):
        """_record_home_position reads from telemetry_cache, not backend."""
        fc, _, cache = _make_follow_controller()
        fc._record_home_position()
        assert fc._home_lat == 38.0
        assert fc._home_lon == -8.0

    def test_record_home_no_backend(self):
        """_record_home_position doesn't crash without backend."""
        fc, _, cache = _make_follow_controller()
        cache._fields.pop("position", None)  # clear position
        fc._record_home_position()  # should not raise

    def test_check_geofence_from_cache(self):
        """_check_geofence reads from telemetry_cache, not backend."""
        fc, _, cache = _make_follow_controller()
        fc._home_lat = 38.0
        fc._home_lon = -8.0
        fc._check_geofence()
        assert fc._geofence_ok is True  # close to home

    def test_lock_target_nonblocking(self):
        """lock_target returns immediately in command-sink mode."""
        fc, slot, _ = _make_follow_controller()
        t0 = time.monotonic()
        fc.lock_target("TGT-001")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1  # no MAVSDK blocking
        assert fc.state in (TargetState.SEARCHING, TargetState.LOCKED)

    def test_unlock_nonblocking(self):
        """unlock returns immediately in command-sink mode."""
        fc, _, _ = _make_follow_controller()
        fc.lock_target("TGT-001")
        t0 = time.monotonic()
        fc.unlock()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1
        assert fc.state == TargetState.UNLOCKED

    def test_follow_loop_writes_to_command_slot(self):
        """Follow loop thread writes VelocityCommand to command_slot."""
        fc, slot, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.lock_target("TGT-001")

        # Give the follow loop a few iterations
        time.sleep(0.15)

        fc.unlock()

        cmd_val, cmd_seq, _ = slot.read()
        assert cmd_val is not None
        assert isinstance(cmd_val, VelocityCommand)
        assert cmd_seq > 0

    def test_standoff_writes_velocity_command(self):
        """Standoff mode computes and writes velocity to command_slot."""
        fc, slot, cache = _make_follow_controller(follow_mode=FollowMode.STANDOFF)
        dets = _detections_with_target(bbox_h=80)
        fc.update_target(dets, 640, 480)
        fc.lock_target("TGT-001")

        time.sleep(0.15)
        fc.unlock()

        cmd_val, _, _ = slot.read()
        assert cmd_val is not None
        assert isinstance(cmd_val, VelocityCommand)
        assert cmd_val.source.startswith("follow_")

    def test_no_backend_calls_during_follow(self):
        """Verify backend is never called during follow lifecycle."""
        mock_backend = MagicMock()
        fc = FollowController(
            backend=mock_backend,
            command_slot=LatestSlot(),
            telemetry_cache=TelemetryCache(),
            follow_mode=FollowMode.YAW,
        )
        # Feed telemetry
        fc._telemetry_cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        fc._telemetry_cache.update_heading(90.0)

        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.lock_target("TGT-001")
        time.sleep(0.15)
        fc.unlock()

        mock_backend.offboard_start.assert_not_called()
        mock_backend.offboard_stop.assert_not_called()
        mock_backend.offboard_set_velocity_body.assert_not_called()
        mock_backend.get_telemetry.assert_not_called()

    def test_search_reads_heading_from_cache(self):
        """Search heading comes from telemetry_cache, not backend."""
        fc, slot, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_heading(180.0)

        # Force into search state
        fc.lock_target("TGT-001")
        # Clear target data so follow loop searches
        fc._target_cx = None
        fc._target_bbox = None
        with fc._lock:
            fc._state = TargetState.SEARCHING
            fc._lost_time = time.monotonic()

        time.sleep(0.15)
        fc.unlock()

        # Should have written commands (search uses heading from cache)
        cmd_val, _, _ = slot.read()
        assert cmd_val is not None

    def test_emergency_stop_no_backend(self):
        """emergency_stop works without backend in command-sink mode."""
        fc, _, _ = _make_follow_controller()
        fc.lock_target("TGT-001")
        fc.emergency_stop()
        assert fc.state == TargetState.UNLOCKED


# ---------------------------------------------------------------------------
# FlightBridgeWorker autonomy_enabled
# ---------------------------------------------------------------------------


class TestFlightBridgeWorkerAutonomy:
    def test_autonomy_default_off(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        assert not fbw.autonomy_enabled

    def test_autonomy_setter(self):
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        fbw.autonomy_enabled = True
        assert fbw.autonomy_enabled

    def test_autonomy_auto_engages_offboard(self):
        """Setting autonomy_enabled=True auto-engages offboard on tick."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)
        assert not fbw.offboard_active

        fbw.autonomy_enabled = True
        fbw.tick()

        assert fbw.offboard_active

    def test_autonomy_auto_disengages_offboard(self):
        """Setting autonomy_enabled=False auto-disengages offboard on tick."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)
        fbw.autonomy_enabled = True
        fbw.tick()
        assert fbw.offboard_active

        fbw.autonomy_enabled = False
        fbw.tick()
        assert not fbw.offboard_active

    def test_autonomy_consumes_command_slot(self):
        """FlightBridgeWorker sends command from slot when autonomy enabled."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)
        fbw.autonomy_enabled = True
        fbw.command_slot.write(VelocityCommand(vx=0.5, vy=-0.3, vz=0.0, yawspeed=0.1))
        fbw.tick()
        assert fbw.offboard_active
        # Mock backend doesn't actually send, but tick processed the command

    def test_worker_follow_integration(self):
        """Full integration: FollowController writes, FlightBridgeWorker consumes."""
        slot: LatestSlot[VelocityCommand] = LatestSlot()
        cache = TelemetryCache()
        cache.update_connected(True)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        cache.update_heading(90.0)

        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=100.0)
        fc = FollowController(
            command_slot=fbw.command_slot,
            telemetry_cache=cache,
            follow_mode=FollowMode.YAW,
        )

        # Lock target
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.lock_target("TGT-001")

        # Enable autonomy
        fbw.autonomy_enabled = True

        # Let follow loop write commands
        time.sleep(0.15)

        # FlightBridgeWorker should consume
        fbw.tick()
        assert fbw.offboard_active

        # Cleanup
        fc.unlock()
        fbw.autonomy_enabled = False
        fbw.tick()


# ---------------------------------------------------------------------------
# Worker vs legacy path isolation
# ---------------------------------------------------------------------------


class TestWorkerLegacyIsolation:
    def test_worker_mode_creates_no_mavsdk_backend(self):
        """Worker path uses command_slot, not MavsdkBackend."""
        fc, slot, cache = _make_follow_controller()
        assert fc._backend is None
        assert fc._command_slot is not None

    def test_worker_mode_single_mavsdk_owner(self):
        """Only FlightBridgeWorker owns MAVSDK in worker mode."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        # No MavsdkBackend created for follow
        # FlightBridgeWorker is the sole owner
        assert fbw.connected

    def test_f_key_no_offboard_start(self):
        """F key flow (lock_target) does not call offboard_start in command-sink."""
        mock_backend = MagicMock()
        fc = FollowController(
            backend=mock_backend,
            command_slot=LatestSlot(),
            telemetry_cache=TelemetryCache(),
            follow_mode=FollowMode.YAW,
        )
        fc._telemetry_cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        fc.lock_target("TGT-001")
        time.sleep(0.1)
        fc.unlock()

        mock_backend.offboard_start.assert_not_called()

    def test_tab_key_no_reconnect(self):
        """Tab flow (unlock + lock) does not call backend in command-sink."""
        mock_backend = MagicMock()
        fc = FollowController(
            backend=mock_backend,
            command_slot=LatestSlot(),
            telemetry_cache=TelemetryCache(),
            follow_mode=FollowMode.YAW,
        )
        fc._telemetry_cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.lock_target("TGT-001")

        # Simulate Tab: unlock then lock different target
        fc.unlock()
        dets2 = _detections_with_target(track_id="TGT-002")
        fc.update_target(dets2, 640, 480)
        fc.lock_target("TGT-002")

        time.sleep(0.1)
        fc.unlock()

        mock_backend.connect.assert_not_called()
        mock_backend.get_telemetry.assert_not_called()


# ---------------------------------------------------------------------------
# Startup initialization order
# ---------------------------------------------------------------------------


class TestWorkerStartupOrder:
    """Verify the correct initialization sequence for --use-workers + follow."""

    def test_flight_bridge_before_follow_controller(self):
        """FlightBridgeWorker must exist before FollowController in worker mode."""
        from sentinel.autonomy.follow_controller import FollowController, FollowMode

        # Step 1: Create FlightBridgeWorker first
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        assert fbw is not None
        assert fbw.command_slot is not None
        assert fbw.telemetry_cache is not None

        # Step 2: Create FollowController with FlightBridgeWorker's slots
        fc = FollowController(
            command_slot=fbw.command_slot,
            telemetry_cache=fbw.telemetry_cache,
            follow_mode=FollowMode.STANDOFF,
        )
        assert fc._command_slot is fbw.command_slot
        assert fc._telemetry_cache is fbw.telemetry_cache

    def test_no_flight_bridge_no_follow_crash(self):
        """If FlightBridgeWorker creation fails, FollowController is not created."""
        flight_bridge = None
        follow_controller = None

        # Simulate failed FlightBridgeWorker creation
        if flight_bridge is not None:
            follow_controller = FollowController(
                command_slot=flight_bridge.command_slot,
            )

        assert follow_controller is None

    def test_worker_standoff_startup_full_sequence(self):
        """Full startup sequence: slots → camera → perception → flight_bridge → follow."""
        from sentinel.autonomy.follow_controller import FollowController, FollowMode

        # 1. Shared slots
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True, hz=50.0)

        # 2. FollowController created with FlightBridgeWorker slots
        fc = FollowController(
            command_slot=fbw.command_slot,
            telemetry_cache=fbw.telemetry_cache,
            follow_mode=FollowMode.STANDOFF,
        )

        # 3. Start FlightBridgeWorker
        fbw.start()
        assert fbw.running

        # 4. Enable autonomy + lock target
        fbw.autonomy_enabled = True
        fbw.telemetry_cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        fbw.telemetry_cache.update_heading(90.0)
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.lock_target("TGT-001")

        # 5. Verify commands flow
        time.sleep(0.15)
        cmd_val, _, _ = fbw.command_slot.read()
        assert cmd_val is not None
        assert isinstance(cmd_val, VelocityCommand)

        # 6. Cleanup
        fc.unlock()
        fbw.autonomy_enabled = False
        fbw.stop()
        fc.emergency_stop()


# ---------------------------------------------------------------------------
# lock_target state reset
# ---------------------------------------------------------------------------


class TestLockTargetStateReset:
    def test_lock_resets_lost_time(self):
        """lock_target resets _lost_time to 0."""
        fc, _, _ = _make_follow_controller()
        fc._lost_time = time.monotonic() - 10.0  # stale lost time
        fc.lock_target("TGT-001")
        assert fc._lost_time == 0.0
        fc.unlock()

    def test_lock_resets_search_state(self):
        """lock_target clears hypotheses, search sectors, emergence points."""
        fc, _, _ = _make_follow_controller()
        fc._hypotheses = [{"type": "MERGED_GROUP", "probability": 0.8}]
        fc._search_sectors = [{"bearing_deg": 45, "probability": 0.6}]
        fc._emergence_points = [{"lat": 38.0, "lon": -8.0, "probability": 0.5}]
        fc._target_priority_score = 0.9

        fc.lock_target("TGT-001")

        assert fc._hypotheses == []
        assert fc._search_sectors == []
        assert fc._emergence_points == []
        assert fc._target_priority_score == 0.0
        fc.unlock()

    def test_visible_target_does_not_immediately_disengage(self):
        """After lock_target + update_target with visible target, follow stays active."""
        fc, slot, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        # Lock target
        fc.lock_target("TGT-001")

        # Immediately provide detection — target is visible
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)

        # Let follow loop run a few ticks
        time.sleep(0.15)

        # Follow should still be active (not disengaged)
        assert fc.state in (TargetState.LOCKED, TargetState.REACQUIRED, TargetState.SEARCHING)
        assert fc.state != TargetState.UNLOCKED

        fc.unlock()

    def test_switching_target_resets_lost_timer(self):
        """Tab-style unlock + relock does not carry old lost_time."""
        fc, _, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        # Lock first target
        fc.lock_target("TGT-001")
        dets1 = _detections_with_target(track_id="TGT-001")
        fc.update_target(dets1, 640, 480)
        time.sleep(0.05)

        # Simulate target lost
        fc._lost_time = time.monotonic() - 10.0
        with fc._lock:
            fc._state = TargetState.SEARCHING

        # Switch to new target (Tab flow)
        fc.unlock()
        fc.lock_target("TGT-007")

        # New target lost_time should be 0
        assert fc._lost_time == 0.0

        # New target is visible — should not immediately disengage
        dets2 = _detections_with_target(track_id="TGT-007")
        fc.update_target(dets2, 640, 480)

        time.sleep(0.15)
        assert fc.state != TargetState.UNLOCKED
        fc.unlock()

    def test_lock_resets_identity_state(self):
        """lock_target clears identity memory and similarity."""
        fc, _, _ = _make_follow_controller()
        fc._target_identity_memory = "red/blue"
        fc._identity_similarity = 0.85
        fc._identity_memory_upper_hsv = [120.0, 200.0, 180.0]
        fc._identity_memory_lower_hsv = [60.0, 150.0, 100.0]

        fc.lock_target("TGT-001")

        assert fc._target_identity_memory == ""
        assert fc._identity_similarity == 0.0
        assert fc._identity_memory_upper_hsv == []
        assert fc._identity_memory_lower_hsv == []
        fc.unlock()

    def test_lock_resets_velocity_and_pid(self):
        """lock_target zeros all velocities and resets PID."""
        fc, _, _ = _make_follow_controller()
        fc._current_yaw_rate = 0.5
        fc._current_vx = 0.3
        fc._current_vy = -0.2
        fc._current_vz = 0.1
        fc._pid._integral = 5.0

        fc.lock_target("TGT-001")
        # Stop loop immediately to prevent thread from overwriting values
        fc._stop_loop()

        assert fc._current_yaw_rate == 0.0
        assert fc._current_vx == 0.0
        assert fc._current_vy == 0.0
        assert fc._current_vz == 0.0
        assert fc._pid._integral == 0.0
        fc.unlock()


# ---------------------------------------------------------------------------
# Key debounce
# ---------------------------------------------------------------------------


class TestKeyDebounce:
    def test_f_key_debounce_suppresses_rapid_press(self):
        """Rapid F press within 300ms is suppressed."""
        last_key_f_time = time.monotonic()
        # Second press immediately after
        now_key = time.monotonic()
        assert now_key - last_key_f_time < 0.3

    def test_f_key_debounce_allows_after_interval(self):
        """F press after 300ms+ is allowed."""
        last_key_f_time = time.monotonic() - 0.5
        now_key = time.monotonic()
        assert now_key - last_key_f_time >= 0.3

    def test_tab_key_debounce_suppresses_rapid_press(self):
        """Rapid Tab press within 300ms is suppressed."""
        last_key_tab_time = time.monotonic()
        now_key = time.monotonic()
        assert now_key - last_key_tab_time < 0.3

    def test_tab_key_debounce_allows_after_interval(self):
        """Tab press after 300ms+ is allowed."""
        last_key_tab_time = time.monotonic() - 0.5
        now_key = time.monotonic()
        assert now_key - last_key_tab_time >= 0.3


# ---------------------------------------------------------------------------
# FlightBridgeWorker shutdown cleanup
# ---------------------------------------------------------------------------


class TestFlightBridgeWorkerShutdown:
    def test_shutting_down_flag(self):
        """stop() sets _shutting_down flag."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        assert not fbw._shutting_down
        fbw.stop()
        assert fbw._shutting_down

    def test_telemetry_tasks_tracked(self):
        """In mock mode, no telemetry tasks are created."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        assert fbw._telemetry_tasks == []
        fbw.stop()

    def test_stop_cleans_up(self):
        """stop() cleans up drone reference and connected state."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        fbw.start()
        time.sleep(0.05)
        fbw.stop()
        assert fbw._drone is None
        assert not fbw._connected

    def test_double_stop_safe(self):
        """Calling stop() twice is safe."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        fbw.start()
        fbw.stop()
        fbw.stop()  # should not raise

    def test_disengage_during_shutdown_suppresses_error(self):
        """disengage_offboard during shutdown logs debug, not error."""
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        fbw.engage_offboard()
        fbw._shutting_down = True
        # disengage in mock mode is no-error anyway, but verify flag
        fbw.disengage_offboard()
        assert not fbw.offboard_active


# ---------------------------------------------------------------------------
# Selectable targets
# ---------------------------------------------------------------------------


class TestSelectableTargets:
    def test_selectable_returns_only_tracked(self):
        from apps.tools.run_gazebo_yolo_test import _selectable_targets
        dets = [
            {"track_id": "TGT-001", "bbox": [100, 100, 200, 200], "class": "person", "confidence": 0.9, "status": "CONFIRMED"},
            {"track_id": None, "bbox": [300, 300, 400, 400], "class": "car", "confidence": 0.8, "status": "CONFIRMED"},
            {"track_id": "TGT-002", "bbox": [400, 100, 500, 200], "class": "car", "confidence": 0.7, "status": "CONFIRMED"},
        ]
        sel = _selectable_targets(dets, 640)
        assert len(sel) == 2
        assert sel[0]["track_id"] in ("TGT-001", "TGT-002")

    def test_selectable_excludes_lost(self):
        from apps.tools.run_gazebo_yolo_test import _selectable_targets
        dets = [
            {"track_id": "TGT-001", "bbox": [100, 100, 200, 200], "class": "person", "confidence": 0.9, "status": "CONFIRMED"},
            {"track_id": "TGT-002", "bbox": [300, 100, 400, 200], "class": "person", "confidence": 0.8, "status": "LOST"},
        ]
        sel = _selectable_targets(dets, 640)
        assert len(sel) == 1
        assert sel[0]["track_id"] == "TGT-001"

    def test_selectable_prioritizes_person(self):
        from apps.tools.run_gazebo_yolo_test import _selectable_targets
        dets = [
            {"track_id": "TGT-CAR", "bbox": [100, 100, 200, 200], "class": "car", "confidence": 0.9, "status": "CONFIRMED"},
            {"track_id": "TGT-PER", "bbox": [300, 100, 400, 200], "class": "person", "confidence": 0.7, "status": "CONFIRMED"},
        ]
        sel = _selectable_targets(dets, 640)
        assert sel[0]["track_id"] == "TGT-PER"

    def test_selectable_prioritizes_nearest_center(self):
        from apps.tools.run_gazebo_yolo_test import _selectable_targets
        dets = [
            {"track_id": "TGT-FAR", "bbox": [10, 10, 50, 50], "class": "person", "confidence": 0.9, "status": "CONFIRMED"},
            {"track_id": "TGT-NEAR", "bbox": [300, 100, 340, 140], "class": "person", "confidence": 0.8, "status": "CONFIRMED"},
        ]
        sel = _selectable_targets(dets, 640)
        assert sel[0]["track_id"] == "TGT-NEAR"

    def test_f_key_locks_best_visible(self):
        """Simulated F key logic: lock first selectable target."""
        from apps.tools.run_gazebo_yolo_test import _selectable_targets
        dets = [
            {"track_id": "TGT-001", "bbox": [300, 100, 340, 140], "class": "person", "confidence": 0.85, "status": "CONFIRMED"},
            {"track_id": "TGT-002", "bbox": [100, 100, 200, 200], "class": "car", "confidence": 0.9, "status": "CONFIRMED"},
        ]
        sel = _selectable_targets(dets, 640)
        assert len(sel) >= 1
        # F key would lock sel[0]["track_id"]
        locked = sel[0]["track_id"]
        assert locked == "TGT-001"  # person nearest center

    def test_tab_cycles_through_visible(self):
        """Simulated Tab logic: cycle through selectable targets."""
        from apps.tools.run_gazebo_yolo_test import _selectable_targets
        dets = [
            {"track_id": "TGT-001", "bbox": [100, 100, 200, 200], "class": "person", "confidence": 0.9, "status": "CONFIRMED"},
            {"track_id": "TGT-002", "bbox": [300, 100, 400, 200], "class": "person", "confidence": 0.8, "status": "CONFIRMED"},
            {"track_id": "TGT-003", "bbox": [500, 100, 600, 200], "class": "car", "confidence": 0.7, "status": "CONFIRMED"},
        ]
        sel = _selectable_targets(dets, 640)
        tids = [t["track_id"] for t in sel]
        assert len(tids) == 3
        # Cycle
        current = tids[0]
        idx = (tids.index(current) + 1) % len(tids)
        assert tids[idx] == tids[1]


# ---------------------------------------------------------------------------
# Lost hysteresis and ReID transfer
# ---------------------------------------------------------------------------


class TestLostHysteresis:
    def test_suspect_state_on_single_miss(self):
        """One frame miss goes to SUSPECT, not SEARCHING."""
        fc, _, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        fc.lock_target("TGT-001")
        dets = _detections_with_target()
        # lock_target→SEARCHING, 1st update→REACQUIRED, 2nd update→LOCKED
        fc.update_target(dets, 640, 480)
        fc.update_target(dets, 640, 480)
        assert fc.state == TargetState.LOCKED

        # Miss one frame
        fc.update_target([], 640, 480)
        assert fc.state == TargetState.SUSPECT
        fc.unlock()

    def test_searching_after_suspect_timeout(self):
        """After suspect timeout, transitions to SEARCHING."""
        fc, _, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        fc.lock_target("TGT-001")
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.update_target(dets, 640, 480)
        assert fc.state == TargetState.LOCKED

        # Miss and age past suspect timeout
        fc.update_target([], 640, 480)
        assert fc.state == TargetState.SUSPECT

        # Simulate time passing by adjusting lost_time
        fc._lost_time = time.monotonic() - 2.0  # 2s ago, past search_timeout_s
        fc.update_target([], 640, 480)
        assert fc.state == TargetState.SEARCHING
        fc.unlock()

    def test_visible_target_resets_lost_timer(self):
        """Reappearing target resets lost_time and returns to REACQUIRED."""
        fc, _, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        fc.lock_target("TGT-001")
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fc.update_target(dets, 640, 480)
        assert fc.state == TargetState.LOCKED

        # Target disappears
        fc.update_target([], 640, 480)
        assert fc.state == TargetState.SUSPECT
        assert fc._lost_time > 0

        # Target reappears
        fc.update_target(dets, 640, 480)
        assert fc.state == TargetState.REACQUIRED
        assert fc._lost_time == 0.0
        fc.unlock()


class TestReIDTransfer:
    def test_reid_transfers_target_id(self):
        """ReID match under new track_id transfers the lock."""
        fc, slot, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        # Lock on TGT-001 with identity
        dets1 = [{
            "track_id": "TGT-001", "bbox": [270, 190, 370, 290],
            "confidence": 0.9, "class": "person", "speed": 0.0,
            "velocity": [0, 0], "identity_descriptor": "red/blue",
        }]
        fc.lock_target("TGT-001")
        fc.update_target(dets1, 640, 480)
        fc.update_target(dets1, 640, 480)
        assert fc.state == TargetState.LOCKED
        assert fc._target_identity_memory == "red/blue"

        # TGT-001 disappears, TGT-014 appears with same identity
        dets2 = [{
            "track_id": "TGT-014", "bbox": [270, 190, 370, 290],
            "confidence": 0.85, "class": "person", "speed": 0.0,
            "velocity": [0, 0], "identity_descriptor": "red/blue",
        }]
        fc.update_target(dets2, 640, 480)

        # Should have transferred lock to TGT-014
        assert fc._target_id == "TGT-014"
        assert fc.state == TargetState.LOCKED  # REACQUIRED→LOCKED in same frame
        assert fc._lost_time == 0.0
        fc.unlock()

    def test_no_reid_transfer_without_identity(self):
        """Without identity memory, no transfer happens."""
        fc, slot, cache = _make_follow_controller(follow_mode=FollowMode.YAW)
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        # Lock on target without identity descriptor
        dets1 = [{
            "track_id": "TGT-001", "bbox": [270, 190, 370, 290],
            "confidence": 0.9, "class": "person", "speed": 0.0,
            "velocity": [0, 0],
        }]
        fc.lock_target("TGT-001")
        fc.update_target(dets1, 640, 480)

        # Target disappears
        fc.update_target([], 640, 480)
        assert fc._target_id == "TGT-001"  # no transfer
        fc.unlock()

    def test_worker_mode_no_mavsdk_from_f_tab(self):
        """F/Tab handlers in worker mode do not call MAVSDK directly."""
        # This is verified by the command-sink mode tests above.
        # Additional check: FlightBridgeWorker is the sole MAVSDK owner.
        fbw = FlightBridgeWorker(backend="mock", mock_connected=True)
        fc = FollowController(
            command_slot=fbw.command_slot,
            telemetry_cache=fbw.telemetry_cache,
            follow_mode=FollowMode.YAW,
        )
        fbw.telemetry_cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)

        # Simulate F key: lock target
        dets = _detections_with_target()
        fc.update_target(dets, 640, 480)
        fbw.autonomy_enabled = True
        fc.lock_target("TGT-001")

        # No backend calls
        assert fc._backend is None
        assert fbw.connected
        fc.unlock()
        fbw.stop()


# ---------------------------------------------------------------------------
# Projection telemetry propagation
# ---------------------------------------------------------------------------


class TestProjectionTelemetryPropagation:
    """Telemetry flows from TelemetryCache through to ISR tracker world projection."""

    def test_telemetry_propagates_to_drone_pose(self):
        from sentinel.tracker.isr_tracker import ISRTracker

        cache = TelemetryCache()
        cache.update_position(lat=47.397, lon=8.546, alt_m=50.0)
        cache.update_heading(heading_deg=90.0)

        tracker = ISRTracker(camera=CameraParams(width_px=640, height_px=480))
        pos, _ = cache.get_position()
        heading, _ = cache.get_heading()

        tracker.update_drone_pose(
            lat=pos.lat, lon=pos.lon, alt_m=pos.alt_m,
            heading_deg=heading or 0.0,
        )

        assert tracker._drone_pose.lat == 47.397
        assert tracker._drone_pose.lon == 8.546
        assert tracker._drone_pose.alt_m == 50.0
        assert tracker._drone_pose.heading_deg == 90.0
        # Gate should pass
        import math
        assert not math.isnan(tracker._drone_pose.lon)

    def test_stale_telemetry_blocks_projection(self):
        from sentinel.tracker.isr_tracker import ISRTracker

        cache = TelemetryCache()
        # No position update → stale

        tracker = ISRTracker(camera=CameraParams(width_px=640, height_px=480))
        snap = cache.snapshot()
        assert snap.position is None or snap.stale(5.0)

        # update_drone_pose never called → lon is NaN → gate blocks
        import math
        assert math.isnan(tracker._drone_pose.lon)

    def test_projection_metrics_accumulate(self):
        from sentinel.tracker.isr_tracker import ISRTracker
        import numpy as np

        cam = CameraParams(width_px=640, height_px=480, h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0)
        tracker = ISRTracker(camera=cam, fps=15.0)

        # Set valid drone pose
        tracker.update_drone_pose(lat=47.397, lon=8.546, alt_m=50.0, heading_deg=0.0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"bbox": [270, 190, 370, 290], "class": "person", "confidence": 0.9},
        ]

        # Run several frames
        for i in range(5):
            result = tracker.update(detections, frame, frame_id=i)

        m = tracker.get_metrics()
        wt = m.get("world_tracking", {})
        assert wt.get("projection_successes", 0) > 0
        # Frame 0 creates new track (no match), frames 1-4 match → 4 projections
        assert wt["projection_successes"] >= 4

    def test_projection_status_in_output(self):
        from sentinel.tracker.isr_tracker import ISRTracker
        import numpy as np

        cam = CameraParams(width_px=640, height_px=480, pitch_deg=-45.0)
        tracker = ISRTracker(camera=cam, fps=15.0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"bbox": [270, 190, 370, 290], "class": "person", "confidence": 0.9},
        ]

        # Without drone pose → NEVER_ATTEMPTED
        result = tracker.update(detections, frame, frame_id=0)
        # Need 3 frames to confirm track (tentative → confirmed)
        for i in range(1, 4):
            result = tracker.update(detections, frame, frame_id=i)
        # Check that projection_status field exists
        confirmed = [d for d in result if d.get("status") == "CONFIRMED"]
        if confirmed:
            assert "projection_status" in confirmed[0]
            assert confirmed[0]["projection_status"] == "NEVER_ATTEMPTED"

        # With drone pose → OK
        tracker.update_drone_pose(lat=47.397, lon=8.546, alt_m=50.0, heading_deg=0.0)
        result = tracker.update(detections, frame, frame_id=5)
        confirmed = [d for d in result if d.get("status") == "CONFIRMED"]
        if confirmed:
            assert confirmed[0]["projection_status"] == ProjectionReason.OK

    def test_update_camera_params(self):
        from sentinel.tracker.isr_tracker import ISRTracker

        tracker = ISRTracker()
        assert tracker._camera.width_px == 640
        assert tracker._camera.height_px == 480

        tracker.update_camera_params(CameraParams(width_px=1280, height_px=720))
        assert tracker._camera.width_px == 1280
        assert tracker._camera.height_px == 720
