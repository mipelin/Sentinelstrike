"""Tests for view_gazebo_camera FrameState and counter logic."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_image_msg(
    width: int = 64,
    height: int = 48,
    channels: int = 3,
    pixel_format: int = 3,
    color: int = 0,
) -> MagicMock:
    msg = MagicMock()
    msg.width = width
    msg.height = height
    msg.step = width * channels
    msg.pixel_format_type = pixel_format
    raw = np.full((height, width, channels), color, dtype=np.uint8)
    msg.data = raw.tobytes()
    msg.header = MagicMock()
    msg.header.stamp = MagicMock()
    msg.header.stamp.sec = 0
    msg.header.stamp.nsec = 0
    return msg


@pytest.fixture(autouse=True)
def _mock_gz(monkeypatch):
    """Patch gz and cv2 imports so the module can load without Gazebo."""
    mock_gz = types.ModuleType("gz")
    mock_transport = types.ModuleType("gz.transport13")
    mock_msgs = types.ModuleType("gz.msgs10")
    mock_image = types.ModuleType("gz.msgs10.image_pb2")

    class _MockNode:
        def subscribe(self, *, msg_type, topic, callback):
            pass

    mock_transport.Node = _MockNode
    mock_image.Image = MagicMock

    mock_gz.transport13 = mock_transport
    mock_gz.msgs10 = mock_msgs
    mock_msgs.image_pb2 = mock_image

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.image_pb2", mock_image)

    for mod in list(sys.modules):
        if "view_gazebo_camera" in mod:
            del sys.modules[mod]


def _get_module():
    return importlib.import_module("apps.tools.view_gazebo_camera")


# ---------------------------------------------------------------------------
# FrameState unit tests
# ---------------------------------------------------------------------------


class TestFrameState:
    def test_initial_state(self):
        mod = _get_module()
        state = mod.FrameState()
        frame, count = state.snapshot()
        assert frame is None
        assert count == 0

    def test_single_callback_increments_count(self):
        mod = _get_module()
        state = mod.FrameState()
        state.on_msg(_make_image_msg(color=42))
        frame, count = state.snapshot()
        assert count == 1
        assert frame is not None
        assert frame.shape == (48, 64, 3)

    def test_ten_callbacks_latest_frame_is_last(self):
        mod = _get_module()
        state = mod.FrameState()
        for i in range(1, 11):
            state.on_msg(_make_image_msg(color=i * 10))

        frame, count = state.snapshot()
        assert count == 10
        # The latest frame should have color from the 10th callback
        assert frame[0, 0, 0] == 100

    def test_snapshot_returns_copy(self):
        mod = _get_module()
        state = mod.FrameState()
        state.on_msg(_make_image_msg(color=50))
        frame1, _ = state.snapshot()
        frame1[0, 0, 0] = 255
        frame2, _ = state.snapshot()
        # Original should not be mutated
        assert frame2[0, 0, 0] == 50

    def test_connected_event_set_on_first_msg(self):
        mod = _get_module()
        state = mod.FrameState()
        assert not state.connected.is_set()
        state.on_msg(_make_image_msg())
        assert state.connected.is_set()

    def test_multithreaded_count_accuracy(self):
        mod = _get_module()
        state = mod.FrameState()
        n = 100

        barrier = threading.Barrier(4)

        def _producer(offset: int) -> None:
            barrier.wait()
            for i in range(n):
                state.on_msg(_make_image_msg(color=(offset + i) % 256))

        threads = [threading.Thread(target=_producer, args=(j * 20,)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        _, count = state.snapshot()
        assert count == n * 4


# ---------------------------------------------------------------------------
# _msg_to_bgr tests
# ---------------------------------------------------------------------------


class TestMsgToBgr:
    def test_rgb_pixel_format(self):
        mod = _get_module()
        msg = _make_image_msg(pixel_format=3, channels=3)
        bgr = mod._msg_to_bgr(msg)
        assert bgr is not None
        assert bgr.shape == (48, 64, 3)

    def test_bgr_pixel_format(self):
        mod = _get_module()
        msg = _make_image_msg(pixel_format=8, channels=3)
        bgr = mod._msg_to_bgr(msg)
        assert bgr is not None
        assert bgr.shape == (48, 64, 3)

    def test_grayscale_pixel_format(self):
        mod = _get_module()
        msg = MagicMock()
        msg.width = 32
        msg.height = 24
        msg.pixel_format_type = 29  # L_INT8
        msg.data = np.zeros((24, 32), dtype=np.uint8).tobytes()
        bgr = mod._msg_to_bgr(msg)
        assert bgr is not None
        assert bgr.shape == (24, 32, 3)  # converted to BGR

    def test_empty_data_returns_none(self):
        mod = _get_module()
        msg = MagicMock()
        msg.width = 0
        msg.height = 0
        msg.data = b""
        assert mod._msg_to_bgr(msg) is None

    def test_unknown_pixel_format_returns_none(self):
        mod = _get_module()
        msg = _make_image_msg(pixel_format=999, channels=3)
        assert mod._msg_to_bgr(msg) is None


# ---------------------------------------------------------------------------
# Headless save-every logic
# ---------------------------------------------------------------------------


class TestHeadlessSaveEvery:
    def test_save_every_saves_correct_frames(self, tmp_path):
        mod = _get_module()
        state = mod.FrameState()

        # Simulate 60 callbacks
        for i in range(1, 61):
            state.on_msg(_make_image_msg(color=i))

        _, count = state.snapshot()
        assert count == 60

        # Verify save-every logic: frames 30, 60 should be saved
        saved: set[int] = set()
        for recv_n in range(1, count + 1):
            if recv_n % 30 == 0 and recv_n not in saved:
                saved.add(recv_n)

        assert 30 in saved
        assert 60 in saved
        assert 29 not in saved
        assert 31 not in saved

    def test_save_every_no_duplicate_saves(self):
        """Reading the same snapshot multiple times must not trigger duplicate saves."""
        mod = _get_module()
        state = mod.FrameState()

        for i in range(1, 61):
            state.on_msg(_make_image_msg(color=i))

        saved_every_set: set[int] = set()
        n = 30
        save_every = 30

        # Simulate reading snapshot multiple times for the same recv_count
        for _ in range(5):
            _, recv_count = state.snapshot()
            if recv_count % save_every == 0 and recv_count not in saved_every_set:
                saved_every_set.add(recv_count)

        assert saved_every_set == {60}

    def test_headless_loop_saves_latest_frame(self, tmp_path):
        mod = _get_module()
        state = mod.FrameState()

        # Simulate 10 callbacks with distinct colors
        for i in range(1, 11):
            state.on_msg(_make_image_msg(color=i * 20))

        frame, count = state.snapshot()
        assert count == 10
        assert frame is not None
        # Latest frame should have color 200 (10 * 20)
        assert frame[0, 0, 0] == 200

        # Save it and verify
        save_path = tmp_path / "latest.jpg"
        import cv2

        cv2.imwrite(str(save_path), frame)
        assert save_path.exists()
        loaded = cv2.imread(str(save_path))
        assert loaded[0, 0, 0] == 200


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_resize_width(self):
        mod = _get_module()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = mod._resize(frame, 320)
        assert resized.shape[1] == 320
        assert resized.shape[0] == 240

    def test_resize_zero_returns_original(self):
        mod = _get_module()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mod._resize(frame, 0)
        assert result.shape == frame.shape

    def test_rotate_180(self):
        mod = _get_module()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[0, 0] = [255, 0, 0]
        rotated = mod._apply_transform(frame, 180, "")
        assert rotated[99, 199, 0] == 255

    def test_flip_horizontal(self):
        mod = _get_module()
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[5, 0] = [255, 0, 0]
        flipped = mod._apply_transform(frame, 0, "h")
        assert flipped[5, 19, 0] == 255
