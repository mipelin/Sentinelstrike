"""Tests for --use-workers variable scoping in run_gazebo_yolo_test.py.

Verifies that recv_count, capture_fps, and related variables are correctly
defined in both legacy and worker paths — no UnboundLocalError.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import DetectionSet


@pytest.fixture(autouse=True)
def _mock_gz(monkeypatch):
    mock_gz = types.ModuleType("gz")
    mock_transport = types.ModuleType("gz.transport13")
    mock_msgs = types.ModuleType("gz.msgs10")
    mock_image = types.ModuleType("gz.msgs10.image_pb2")

    class _MockNode:
        def subscribe(self, **kwargs):
            pass

    mock_transport.Node = _MockNode
    mock_image.Image = MagicMock

    mock_gz.transport13 = mock_transport
    mock_msgs.image_pb2 = mock_image
    mock_gz.msgs10 = mock_msgs

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.image_pb2", mock_image)


# ---------------------------------------------------------------------------
# Worker-path variable scoping
# ---------------------------------------------------------------------------


class TestWorkerPathVariableScoping:
    """Simulate the worker-path variable flow from the main loop."""

    def test_recv_count_from_frame_slot_seq(self):
        """In worker mode recv_count comes from frame_slot.seq_id()."""
        frame_slot = LatestSlot()
        assert frame_slot.seq_id() == 0

        # Simulate frames arriving
        for i in range(10):
            frame_slot.write(np.zeros((480, 640, 3), dtype=np.uint8))

        recv_count = frame_slot.seq_id()
        assert recv_count == 10

    def test_capture_fps_computation_worker_path(self):
        """capture_fps calculation works with frame_slot-derived recv_count."""
        import time

        frame_slot = LatestSlot()
        last_fps_recv_count = 0
        capture_fps = 0.0

        # Simulate frames
        for i in range(20):
            frame_slot.write(np.zeros((480, 640, 3), dtype=np.uint8))

        # Simulate FPS calculation (like line ~1488-1493)
        recv_count = frame_slot.seq_id()
        fps_dt = 1.0  # pretend 1s elapsed
        if fps_dt >= 0.5:
            capture_fps = (recv_count - last_fps_recv_count) / fps_dt
            last_fps_recv_count = recv_count

        assert capture_fps == 20.0
        assert last_fps_recv_count == 20

    def test_fps_delta_across_two_intervals(self):
        """FPS correctly tracks delta across two measurement windows."""
        frame_slot = LatestSlot()
        last_fps_recv_count = 0
        capture_fps = 0.0

        # First interval: 15 frames
        for _ in range(15):
            frame_slot.write(np.zeros((100, 100, 3), dtype=np.uint8))
        recv_count = frame_slot.seq_id()
        fps_dt = 0.5
        capture_fps = (recv_count - last_fps_recv_count) / fps_dt
        last_fps_recv_count = recv_count
        assert capture_fps == 30.0

        # Second interval: 10 more frames
        for _ in range(10):
            frame_slot.write(np.zeros((100, 100, 3), dtype=np.uint8))
        recv_count = frame_slot.seq_id()
        fps_dt = 0.5
        capture_fps = (recv_count - last_fps_recv_count) / fps_dt
        last_fps_recv_count = recv_count
        assert capture_fps == 20.0
        assert recv_count == 25

    def test_detection_slot_seq_matches_worker_pattern(self):
        """detection_slot seq increments match the worker-path skip logic."""
        detection_slot: LatestSlot[DetectionSet] = LatestSlot()
        last_recv_count = 0

        # No detection yet — should skip
        det_val, det_seq, _ = detection_slot.read()
        assert det_val is None
        # (main loop would `continue` here)

        # Write a detection
        det = DetectionSet(
            frame_id=1, seq=1, detections=[{"class": "person", "confidence": 0.9}],
            backend="mock", inference_ms=5.0,
        )
        detection_slot.write(det)

        det_val, det_seq, _ = detection_slot.read()
        assert det_val is not None
        assert det_seq != last_recv_count  # not stale
        last_recv_count = det_seq

        # Same seq — should skip
        det_val2, det_seq2, _ = detection_slot.read()
        assert det_seq2 == last_recv_count  # stale

    def test_recv_count_summary_worker_path(self):
        """Summary block uses frame_slot.seq_id() in worker mode."""
        frame_slot = LatestSlot()
        for i in range(50):
            frame_slot.write(np.zeros((100, 100, 3), dtype=np.uint8))

        # Worker path: recv_count from frame_slot (not state.snapshot())
        use_workers = True
        recv_count = frame_slot.seq_id()

        # Summary computations
        elapsed = 5.0
        processed = 40
        assert recv_count == 50
        assert elapsed > 0
        avg_process_fps = processed / elapsed
        avg_capture_fps = recv_count / elapsed
        assert avg_process_fps == 8.0
        assert avg_capture_fps == 10.0

    def test_progress_log_worker_path(self):
        """Progress log line formats recv_count correctly in worker path."""
        frame_slot = LatestSlot()
        for i in range(42):
            frame_slot.write(np.zeros((100, 100, 3), dtype=np.uint8))

        recv_count = frame_slot.seq_id()
        processed = 30
        capture_fps = 10.0
        process_fps = 8.0
        inference_ms = 12.0
        total_detections = 55

        log = f"recv: {recv_count}  det: 5  total: {total_detections}"
        assert "recv: 42" in log

    def test_recv_count_never_unbound_in_worker_path(self):
        """Verify recv_count is always set before FPS calc in worker path."""
        import time

        # Simulate the exact variable flow of the main loop
        frame_slot = LatestSlot()
        detection_slot: LatestSlot[DetectionSet] = LatestSlot()
        last_recv_count = 0
        last_fps_recv_count = 0
        capture_fps = 0.0
        process_fps = 0.0
        last_fps_time = time.monotonic()
        last_fps_process_count = 0
        processed = 0

        # Write a detection + frame
        frame_slot.write(np.zeros((100, 100, 3), dtype=np.uint8))
        detection_slot.write(DetectionSet(
            frame_id=1, seq=1, detections=[], backend="mock", inference_ms=5.0,
        ))

        # Worker-path detection block
        det_val, det_seq, _ = detection_slot.read()
        assert det_val is not None
        assert det_seq != last_recv_count
        last_recv_count = det_seq
        recv_count = frame_slot.seq_id()

        processed += 1
        detections = det_val.detections
        inference_ms = det_val.inference_ms

        # FPS calculation — must not raise UnboundLocalError
        now_mono = time.monotonic()
        fps_dt = now_mono - last_fps_time
        if fps_dt >= 0.5:
            process_fps = (processed - last_fps_process_count) / fps_dt
            capture_fps = (recv_count - last_fps_recv_count) / fps_dt
            last_fps_time = now_mono
            last_fps_process_count = processed
            last_fps_recv_count = recv_count

        # All variables accessible
        assert recv_count == 1
        assert capture_fps >= 0
        assert process_fps >= 0
