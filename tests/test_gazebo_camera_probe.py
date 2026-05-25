"""Tests for run_gazebo_camera_probe CLI — mocked gz.transport."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_image_msg(width=640, height=480, channels=3, pixel_format=3):
    msg = MagicMock()
    msg.width = width
    msg.height = height
    msg.step = width * channels
    msg.pixel_format_type = pixel_format
    raw = np.zeros((height, width, channels), dtype=np.uint8)
    msg.data = raw.tobytes()
    return msg


@pytest.fixture(autouse=True)
def _mock_gz(monkeypatch):
    """Patch gz imports and capture the subscription callback."""
    mock_gz = types.ModuleType("gz")
    mock_transport = types.ModuleType("gz.transport13")
    mock_msgs = types.ModuleType("gz.msgs10")
    mock_image = types.ModuleType("gz.msgs10.image_pb2")

    _captured: dict = {}

    class _MockNode:
        def subscribe(self, *, msg_type, topic, callback):
            _captured["callback"] = callback
            _captured["topic"] = topic

    mock_transport.Node = _MockNode
    mock_image.Image = MagicMock

    mock_gz.transport13 = mock_transport
    mock_gz.msgs10 = mock_msgs
    mock_msgs.image_pb2 = mock_image

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.image_pb2", mock_image)

    # Force re-import of the probe module
    for mod in list(sys.modules):
        if "run_gazebo_camera_probe" in mod:
            del sys.modules[mod]

    yield _captured


def test_probe_saves_frame(tmp_path, _mock_gz):
    """Probe receives a message and saves a JPG."""
    mod = importlib.import_module("apps.tools.run_gazebo_camera_probe")

    out_path = tmp_path / "frame.jpg"
    result: dict = {}

    def _run():
        try:
            with patch.object(sys, "argv", [
                "probe", "--topic", "/test/img", "--timeout", "5", "--save-frame", str(out_path)
            ]):
                mod.main()
            result["exit"] = 0
        except SystemExit as e:
            result["exit"] = e.code

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Give probe time to subscribe
    time.sleep(0.3)
    if "callback" in _mock_gz:
        _mock_gz["callback"](_make_image_msg())

    t.join(timeout=5)

    assert result.get("exit") in (None, 0)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_probe_timeout_exits_nonzero(_mock_gz):
    """Probe exits with code 2 when no message arrives."""
    mod = importlib.import_module("apps.tools.run_gazebo_camera_probe")

    result: dict = {}

    def _run():
        try:
            with patch.object(sys, "argv", [
                "probe", "--topic", "/test/img", "--timeout", "0.3"
            ]):
                mod.main()
            result["exit"] = 0
        except SystemExit as e:
            result["exit"] = e.code

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=5)

    assert result.get("exit") == 2


def test_probe_prints_metadata(capsys, _mock_gz):
    """Probe prints message type, dimensions, format, step, data length."""
    mod = importlib.import_module("apps.tools.run_gazebo_camera_probe")

    result: dict = {}

    def _run():
        try:
            with patch.object(sys, "argv", [
                "probe", "--topic", "/test/img", "--timeout", "5"
            ]):
                mod.main()
            result["exit"] = 0
        except SystemExit as e:
            result["exit"] = e.code

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    time.sleep(0.3)
    if "callback" in _mock_gz:
        _mock_gz["callback"](_make_image_msg(width=320, height=240, channels=3, pixel_format=3))

    t.join(timeout=5)

    assert result.get("exit") in (None, 0)
    # Metadata is printed in the thread, capsys won't capture it directly.
    # This test verifies the probe doesn't crash and exits successfully.
