"""Tests for CameraWorker and PerceptionWorker."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import DetectionSet, FrameSnapshot
from sentinel.runtime.workers.camera_worker import CameraWorker, _msg_to_bgr
from sentinel.runtime.workers.perception_worker import PerceptionWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_fake_msg(
    h: int = 480, w: int = 640, fmt: int = 8,
) -> SimpleNamespace:
    """Create a minimal Gazebo Image-like message for testing."""
    channels = {8: 3, 3: 3, 9: 3, 29: 1}.get(fmt, 3)
    dtype = np.uint8
    data = np.random.randint(0, 255, (h, w, channels), dtype=dtype).tobytes()
    return SimpleNamespace(
        width=h,
        height=w,  # intentionally swapped to match Gazebo convention
        step=w * channels,
        pixel_format_type=fmt,
        data=data,
    )


# ---------------------------------------------------------------------------
# CameraWorker
# ---------------------------------------------------------------------------


class TestCameraWorker:
    def test_initial_state(self):
        cw = CameraWorker(topic="/test/topic")
        assert not cw.connected
        frame, seq, ts = cw.frame_slot.read()
        assert frame is None
        assert seq == 0
        meta, _, _ = cw.meta_slot.read()
        assert meta is None

    def test_on_msg_publishes_frame_and_meta(self):
        cw = CameraWorker(topic="/test/topic")
        msg = _make_fake_msg(480, 640, fmt=8)
        cw._on_msg(msg)

        assert cw.connected
        frame, seq, ts = cw.frame_slot.read()
        assert frame is not None
        assert seq == 1

        meta, _, _ = cw.meta_slot.read()
        assert isinstance(meta, FrameSnapshot)
        assert meta.seq == 1
        assert meta.frame_id == 1
        assert meta.shape is not None

    def test_overwrite_stale_frames(self):
        cw = CameraWorker(topic="/test/topic")

        # Write first frame
        cw._on_msg(_make_fake_msg(480, 640, fmt=8))
        frame1, seq1, _ = cw.frame_slot.read()
        assert seq1 == 1

        # Write second frame — should overwrite
        cw._on_msg(_make_fake_msg(480, 640, fmt=8))
        frame2, seq2, _ = cw.frame_slot.read()
        assert seq2 == 2
        # Only latest frame is kept (no queue, no FIFO)
        assert cw.frame_slot.seq_id() == 2

    def test_frame_seq_increments(self):
        cw = CameraWorker(topic="/test/topic")
        for i in range(10):
            cw._on_msg(_make_fake_msg(480, 640, fmt=8))
        assert cw.frame_slot.seq_id() == 10
        assert cw.meta_slot.seq_id() == 10

    def test_empty_message_ignored(self):
        cw = CameraWorker(topic="/test/topic")
        empty_msg = SimpleNamespace(width=0, height=0, step=0, pixel_format_type=8, data=b"")
        cw._on_msg(empty_msg)
        assert not cw.connected
        assert cw.frame_slot.seq_id() == 0

    def test_start_stop_lifecycle(self):
        cw = CameraWorker(topic="/test/topic")
        cw._connected.set()  # Pretend connected
        cw.start()
        assert cw.running
        time.sleep(0.15)
        cw.stop()
        assert not cw.running

    def test_independent_timing(self):
        """CameraWorker frame arrival doesn't depend on worker thread."""
        cw = CameraWorker(topic="/test/topic")
        cw.start()
        time.sleep(0.05)

        # Simulate frames arriving from Gazebo transport thread
        for _ in range(20):
            cw._on_msg(_make_fake_msg(480, 640, fmt=8))
            time.sleep(0.01)

        assert cw.frame_slot.seq_id() == 20
        cw.stop()


# ---------------------------------------------------------------------------
# PerceptionWorker
# ---------------------------------------------------------------------------


class TestPerceptionWorker:
    def test_skips_when_no_frame(self):
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )
        pw.tick()
        assert pw.detection_slot.seq_id() == 0
        assert pw.metrics.skipped_ticks == 1

    def test_processes_new_frame(self):
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )

        frame = _make_fake_frame()
        frame_slot.write(frame)
        meta_slot.write(FrameSnapshot(frame_id=1, seq=1, shape=frame.shape))

        pw.tick()
        assert pw.detection_slot.seq_id() == 1

        det_val, det_seq, _ = pw.detection_slot.read()
        assert isinstance(det_val, DetectionSet)
        assert len(det_val.detections) == 2
        assert det_val.backend == "mock"
        assert det_val.frame_id == 1

    def test_skips_stale_frame(self):
        """PerceptionWorker must not reprocess the same frame."""
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )

        frame = _make_fake_frame()
        frame_slot.write(frame)
        meta_slot.write(FrameSnapshot(frame_id=1, seq=1, shape=frame.shape))

        # First tick processes the frame
        pw.tick()
        assert pw.detection_slot.seq_id() == 1

        # Second tick sees same seq — should skip
        pw.tick()
        assert pw.detection_slot.seq_id() == 1  # unchanged
        assert pw.metrics.skipped_ticks == 1

    def test_newest_frame_only(self):
        """When multiple frames arrive, only the latest is processed."""
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )

        # Write 5 frames rapidly (simulates fast camera)
        for i in range(5):
            frame_slot.write(_make_fake_frame())
            meta_slot.write(FrameSnapshot(frame_id=i + 1, seq=i + 1))

        # One tick — should only process the latest (seq=5)
        pw.tick()
        det_val, det_seq, _ = pw.detection_slot.read()
        assert det_val.frame_id == 5
        assert det_seq == 1  # detection slot written once

    def test_inference_slowdown_does_not_block_ingest(self):
        """Camera ingest continues even if perception is slow."""
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )

        # Simulate 10 frames arriving from camera
        for i in range(10):
            frame_slot.write(_make_fake_frame())
            meta_slot.write(FrameSnapshot(frame_id=i + 1, seq=i + 1))

        # Camera has received 10 frames
        assert frame_slot.seq_id() == 10

        # Perception processes only the latest
        pw.tick()
        det_val, _, _ = pw.detection_slot.read()
        assert det_val.frame_id == 10  # latest, not first

        # Only one detection published — 9 intermediate frames were never queued
        assert pw.detection_slot.seq_id() == 1

    def test_worker_recovery_after_exceptions(self):
        """Worker continues after inference exceptions."""
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        class FailingPerception(PerceptionWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._call_count = 0

            def _run_inference(self, frame):
                self._call_count += 1
                if self._call_count <= 2:
                    raise RuntimeError("inference crash")
                return super()._run_inference(frame)

        pw = FailingPerception(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )
        pw.start()

        # Feed frames
        for i in range(5):
            frame_slot.write(_make_fake_frame())
            meta_slot.write(FrameSnapshot(frame_id=i + 1, seq=i + 1))
            time.sleep(0.04)

        pw.stop()

        # Worker should have recovered — at least one successful detection
        assert pw.metrics.exceptions >= 2
        assert pw.detection_slot.seq_id() >= 1  # at least one success after recovery

    def test_mock_detection_format(self):
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )

        frame = _make_fake_frame(480, 640)
        frame_slot.write(frame)
        meta_slot.write(FrameSnapshot(frame_id=1, seq=1, shape=frame.shape))

        pw.tick()
        det_val, _, _ = pw.detection_slot.read()
        assert len(det_val.detections) == 2
        person = det_val.detections[0]
        assert person["class"] == "person"
        assert "bbox" in person
        assert "confidence" in person

    def test_independent_worker_timing(self):
        """Camera and perception workers run independently."""
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        # Feed frames from a simulated camera thread
        def feed_frames():
            for i in range(20):
                frame_slot.write(_make_fake_frame())
                meta_slot.write(FrameSnapshot(frame_id=i + 1, seq=i + 1))
                time.sleep(0.02)

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=30.0,
        )
        pw.start()

        feeder = threading.Thread(target=feed_frames)
        feeder.start()
        feeder.join(timeout=2.0)
        time.sleep(0.2)

        pw.stop()

        # Camera ingested all 20 frames
        assert frame_slot.seq_id() == 20
        # Perception processed at least some
        assert pw.detection_slot.seq_id() >= 1
        assert pw.metrics.total_ticks >= 1

    def test_start_stop_lifecycle(self):
        frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

        pw = PerceptionWorker(
            frame_slot=frame_slot,
            meta_slot=meta_slot,
            backend="mock",
            hz=50.0,
        )
        assert not pw.running
        pw.start()
        assert pw.running
        pw.stop()
        assert not pw.running


# ---------------------------------------------------------------------------
# _msg_to_bgr
# ---------------------------------------------------------------------------


class TestMsgToBgr:
    def test_bgr_format(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
        msg = SimpleNamespace(width=10, height=10, step=30, pixel_format_type=8, data=data)
        result = _msg_to_bgr(msg)
        assert result is not None
        assert result.shape == (10, 10, 3)

    def test_empty_returns_none(self):
        msg = SimpleNamespace(width=0, height=0, step=0, pixel_format_type=8, data=b"")
        assert _msg_to_bgr(msg) is None

    def test_unknown_format_returns_none(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8).tobytes()
        msg = SimpleNamespace(width=10, height=10, step=30, pixel_format_type=999, data=data)
        assert _msg_to_bgr(msg) is None
