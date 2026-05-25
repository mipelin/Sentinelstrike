"""Tests for record_gazebo_camera stamping and recording logic."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _mock_gz(monkeypatch):
    """Patch gz imports and cv2 so the module loads without Gazebo."""
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
    mock_gz.msgs10 = mock_msgs
    mock_msgs.image_pb2 = mock_image

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.image_pb2", mock_image)

    # Ensure view_gazebo_camera is also importable (it's needed by record_gazebo_camera)
    for mod in list(sys.modules):
        if "record_gazebo_camera" in mod or "view_gazebo_camera" in mod:
            del sys.modules[mod]


def _get_module():
    return importlib.import_module("apps.tools.record_gazebo_camera")


class TestFrameStamping:
    def test_stamp_adds_overlay(self):
        mod = _get_module()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stamped = mod._stamp_frame(frame, 42, 30.0, 5.5)
        assert stamped is not None
        assert stamped.shape == frame.shape
        # The stamped frame should have non-zero pixels (the text overlay)
        assert stamped.sum() > 0

    def test_stamp_preserves_size(self):
        mod = _get_module()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        stamped = mod._stamp_frame(frame, 100, 15.0, 30.0)
        assert stamped.shape == (720, 1280, 3)

    def test_stamp_does_not_mutate_original(self):
        mod = _get_module()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        original_sum = frame.sum()
        mod._stamp_frame(frame, 1, 10.0, 0.0)
        assert frame.sum() == original_sum


class TestRecordingWithSyntheticFrames:
    def test_video_writer_creates_file(self, tmp_path):
        import cv2

        out_path = str(tmp_path / "test.mp4")
        h, w = 240, 320
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))

        for i in range(30):
            frame = np.full((h, w, 3), i * 8, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

    def test_jpeg_sequence(self, tmp_path):
        import cv2

        jpeg_dir = tmp_path / "frames"
        jpeg_dir.mkdir()

        for i in range(10):
            frame = np.full((120, 160, 3), i * 25, dtype=np.uint8)
            cv2.imwrite(str(jpeg_dir / f"frame_{i:06d}.jpg"), frame)

        jpegs = list(jpeg_dir.glob("*.jpg"))
        assert len(jpegs) == 10
        # Verify frames are readable
        for jp in sorted(jpegs):
            img = cv2.imread(str(jp))
            assert img is not None
            assert img.shape == (120, 160, 3)

    def test_stamped_frame_writeable_to_jpeg(self, tmp_path):
        mod = _get_module()
        import cv2

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        stamped = mod._stamp_frame(frame, 5, 25.0, 1.0)

        out = str(tmp_path / "stamped.jpg")
        cv2.imwrite(out, stamped)
        loaded = cv2.imread(out)
        assert loaded is not None
        assert loaded.shape == (240, 320, 3)
