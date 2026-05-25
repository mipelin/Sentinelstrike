"""Tests for sentinel/runtime infrastructure — blackboard, contracts, metrics, worker."""

from __future__ import annotations

import math
import threading
import time

import pytest

from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import (
    DetectionSet,
    DroneState,
    FlightStatus,
    FrameSnapshot,
    TrackSet,
    VelocityCommand,
    WorldModelSnapshot,
)
from sentinel.runtime.metrics import WorkerMetrics
from sentinel.runtime.worker import Worker, WorkerHealth


# ---------------------------------------------------------------------------
# LatestSlot
# ---------------------------------------------------------------------------


class TestLatestSlot:
    def test_write_read_basic(self):
        slot: LatestSlot[int] = LatestSlot()
        seq = slot.write(42)
        assert seq == 1
        val, seq_out, ts = slot.read()
        assert val == 42
        assert seq_out == 1
        assert ts > 0

    def test_seq_increments(self):
        slot: LatestSlot[str] = LatestSlot()
        assert slot.write("a") == 1
        assert slot.write("b") == 2
        assert slot.write("c") == 3

    def test_overwrite_semantics(self):
        slot: LatestSlot[str] = LatestSlot()
        slot.write("alpha")
        slot.write("beta")
        val, _, _ = slot.read()
        assert val == "beta"

    def test_initial_state(self):
        slot: LatestSlot[int] = LatestSlot()
        val, seq, ts = slot.read()
        assert val is None
        assert seq == 0
        assert ts == 0.0
        assert slot.age_s() == float("inf")
        assert slot.is_stale(1.0) is True
        assert slot.seq_id() == 0

    def test_age_and_staleness(self):
        slot: LatestSlot[int] = LatestSlot()
        slot.write(1)
        time.sleep(0.05)
        age = slot.age_s()
        assert age >= 0.04
        assert slot.is_stale(0.01) is True
        assert slot.is_stale(10.0) is False

    def test_thread_safety(self):
        slot: LatestSlot[int] = LatestSlot()
        writes_per_thread = 500
        n_threads = 4

        def writer(start: int) -> None:
            for i in range(writes_per_thread):
                slot.write(start + i)

        threads = [threading.Thread(target=writer, args=(t * 1000,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = writes_per_thread * n_threads
        assert slot.seq_id() == expected_total
        val, seq, _ = slot.read()
        assert seq == expected_total
        assert isinstance(val, int)

    def test_none_value_is_valid(self):
        slot: LatestSlot[int | None] = LatestSlot()
        slot.write(None)
        val, seq, _ = slot.read()
        assert val is None
        assert seq == 1
        slot.write(42)
        val, seq, _ = slot.read()
        assert val == 42
        assert seq == 2
        slot.write(None)
        val, seq, _ = slot.read()
        assert val is None
        assert seq == 3


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestContracts:
    def test_frame_snapshot_defaults(self):
        f = FrameSnapshot()
        assert f.frame_id == 0
        assert f.seq == 0
        assert f.ts_monotonic == 0.0
        assert f.ts_utc == ""
        assert f.shape is None

    def test_frame_snapshot_with_values(self):
        f = FrameSnapshot(frame_id=5, seq=3, shape=(480, 640, 3))
        assert f.shape == (480, 640, 3)
        assert f.frame_id == 5

    def test_detection_set_with_detections(self):
        ds = DetectionSet(
            frame_id=10,
            detections=[
                {"class": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4]},
                {"class": "car", "confidence": 0.7, "bbox": [5, 6, 7, 8]},
            ],
            backend="yolo",
            inference_ms=12.5,
        )
        assert len(ds.detections) == 2
        assert ds.backend == "yolo"

    def test_track_set_counts(self):
        ts = TrackSet(active_count=3, lost_count=1, tracker_type="isr")
        assert ts.active_count == 3
        assert ts.lost_count == 1

    def test_drone_state_defaults(self):
        d = DroneState()
        assert d.connected is False
        assert d.mode == "UNKNOWN"
        assert d.position is None

    def test_velocity_command(self):
        vc = VelocityCommand(vx=1.0, vy=-0.5, vz=0, yawspeed=0.3, source="follow")
        assert vc.vx == 1.0
        assert vc.source == "follow"

    def test_flight_status_states(self):
        fs = FlightStatus(state="IN_FLIGHT", offboard_active=True, follow_mode="standoff")
        assert fs.state == "IN_FLIGHT"
        assert fs.offboard_active is True

    def test_serialization_roundtrip(self):
        original = DroneState(
            seq=5, ts_utc="2026-01-01T00:00:00Z", connected=True,
            position={"lat": 38.0, "lon": -8.0, "alt_m": 50.0},
        )
        data = original.model_dump()
        restored = DroneState.model_validate(data)
        assert restored == original

    def test_world_model_snapshot(self):
        wm = WorldModelSnapshot(
            hypotheses=[{"type": "OCCLUDED", "probability": 0.8}],
            behavior_events=[{"pattern": "loitering"}],
        )
        assert len(wm.hypotheses) == 1
        assert len(wm.behavior_events) == 1


# ---------------------------------------------------------------------------
# WorkerMetrics
# ---------------------------------------------------------------------------


class TestWorkerMetrics:
    def test_record_tick(self):
        m = WorkerMetrics(name="test")
        for _ in range(5):
            m.record_tick(0.001)
        assert m.total_ticks == 5
        assert m.tick_ms_p50 > 0

    def test_percentiles(self):
        m = WorkerMetrics(name="test")
        for ms in [1, 2, 3, 4, 5]:
            m.record_tick(ms / 1000.0)
        assert abs(m.tick_ms_p50 - 3.0) < 0.5
        assert m.tick_ms_p95 >= 4.0
        assert m.tick_ms_p99 >= 4.0

    def test_empty_percentiles(self):
        m = WorkerMetrics(name="test")
        assert m.tick_ms_p50 == 0.0
        assert m.tick_ms_p95 == 0.0
        assert m.tick_ms_p99 == 0.0

    def test_record_skip(self):
        m = WorkerMetrics(name="test")
        for _ in range(3):
            m.record_skip()
        assert m.skipped_ticks == 3

    def test_record_stale_read(self):
        m = WorkerMetrics(name="test")
        for _ in range(5):
            m.record_stale_read()
        assert m.stale_reads == 5

    def test_record_exception(self):
        m = WorkerMetrics(name="test")
        m.record_exception()
        m.record_exception()
        assert m.exceptions == 2

    def test_last_tick_age_before_any_tick(self):
        m = WorkerMetrics(name="test")
        assert m.last_tick_age_s() == float("inf")

    def test_last_tick_age_after_tick(self):
        m = WorkerMetrics(name="test")
        m.record_tick(0.001)
        time.sleep(0.05)
        assert m.last_tick_age_s() >= 0.04

    def test_snapshot_keys(self):
        m = WorkerMetrics(name="test")
        m.record_tick(0.001)
        snap = m.snapshot()
        expected_keys = {
            "name", "total_ticks", "skipped_ticks", "exceptions",
            "stale_reads", "tick_ms_p50", "tick_ms_p95", "tick_ms_p99",
            "loop_hz", "last_tick_age_s",
        }
        assert set(snap.keys()) == expected_keys

    def test_loop_hz(self):
        m = WorkerMetrics(name="test")
        for _ in range(10):
            m.record_tick(0.001)
            time.sleep(0.01)
        hz = m.loop_hz
        assert 50 <= hz <= 200

    def test_window_size_limit(self):
        m = WorkerMetrics(name="test", )
        for _ in range(150):
            m.record_tick(0.001)
        assert len(m._tick_times) <= 100
        assert len(m._wall_times) <= 100
        assert m.total_ticks == 150


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _CountingWorker(Worker):
    """Test worker that counts ticks and optionally injects failures."""

    def __init__(self, name: str = "test_worker", hz: float = 50.0) -> None:
        super().__init__(name=name, hz=hz)
        self.tick_count = 0
        self._fail_every_n: int = 0

    def tick(self) -> None:
        self.tick_count += 1
        if self._fail_every_n > 0 and self.tick_count % self._fail_every_n == 0:
            raise RuntimeError("injected failure")


class TestWorker:
    def test_start_stop_lifecycle(self):
        w = _CountingWorker(hz=50.0)
        assert not w.running
        w.start()
        assert w.running
        time.sleep(0.15)
        assert w.tick_count > 0
        w.stop()
        assert not w.running

    def test_tick_counting(self):
        w = _CountingWorker(hz=100.0)
        w.start()
        time.sleep(0.3)
        w.stop()
        assert w.metrics.total_ticks >= 10

    def test_exception_isolation(self):
        w = _CountingWorker(hz=100.0)
        w._fail_every_n = 3
        w.start()
        time.sleep(0.3)
        w.stop()
        assert w.metrics.exceptions > 0
        assert w.metrics.total_ticks > 0

    def test_double_start_noop(self):
        w = _CountingWorker(hz=50.0)
        w.start()
        t1 = w._thread
        w.start()  # no-op
        t2 = w._thread
        assert t1 is t2
        w.stop()

    def test_health_report(self):
        w = _CountingWorker(hz=50.0)
        w.start()
        time.sleep(0.1)
        h = w.health()
        assert isinstance(h, WorkerHealth)
        assert h.name == "test_worker"
        assert h.running is True
        assert h.stale is False
        assert "total_ticks" in h.metrics
        w.stop()

    def test_stale_detection(self):
        w = _CountingWorker(hz=50.0)
        w.start()
        time.sleep(0.1)
        w.stop()
        time.sleep(0.15)  # stale threshold = 3/50 = 0.06s
        h = w.health()
        assert h.stale is True

    def test_stop_joins_thread(self):
        w = _CountingWorker(hz=50.0)
        w.start()
        assert w._thread is not None
        w.stop()
        assert w._thread is None
