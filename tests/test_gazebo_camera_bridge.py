"""Tests for GazeboCameraBridge — mocked gz.transport subscription."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_image_msg(width=640, height=480, channels=3, pixel_format=3):
    """Create a mock gz.msgs10.Image with RGB_INT8 data."""
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
    """Patch gz imports at module level before gazebo_camera_bridge is imported."""
    mock_gz = types.ModuleType("gz")
    mock_transport = types.ModuleType("gz.transport13")
    mock_msgs = types.ModuleType("gz.msgs10")
    mock_image = types.ModuleType("gz.msgs10.image_pb2")

    mock_transport.Node = MagicMock
    mock_image.Image = MagicMock

    mock_gz.transport13 = mock_transport
    mock_gz.msgs10 = mock_msgs
    mock_msgs.image_pb2 = mock_image

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.image_pb2", mock_image)

    # Force re-import so mocked modules are picked up
    if "sentinel.sensors.gazebo_camera_bridge" in sys.modules:
        del sys.modules["sentinel.sensors.gazebo_camera_bridge"]

    yield


def test_bridge_construction():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic")
    assert bridge._topic == "/test/topic"
    assert not bridge.health.connected


def test_bridge_open_subscribes():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic")
    result = bridge.open()
    assert result is True
    assert bridge.health.connected


def test_bridge_read_frame_waits_for_first_frame():
    """read_frame should block up to startup_timeout_s instead of returning immediately."""
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.3)
    bridge.open()

    # read_frame should wait then return False when no frame arrives within timeout
    ok, frame = bridge.read_frame()
    assert ok is False
    assert frame is None


def test_bridge_read_frame_gets_frame_with_startup_wait():
    """read_frame should wait and pick up a frame that arrives during startup."""
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=2.0)
    bridge.open()

    msg = _make_image_msg(width=160, height=120, channels=3, pixel_format=3)

    def _deliver():
        time.sleep(0.1)
        bridge._on_image(msg)

    t = threading.Thread(target=_deliver, daemon=True)
    t.start()

    ok, frame = bridge.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
    t.join()


def test_bridge_no_eof_on_startup_timeout():
    """After startup timeout, the bridge is still connected — not EOF."""
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    bridge.open()

    ok, frame = bridge.read_frame()
    assert ok is False
    # Bridge is still connected (no frame yet, but not closed)
    assert bridge.health.connected is True


def test_bridge_read_frame_converts_protobuf_rgb():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    msg = _make_image_msg(width=160, height=120, channels=3, pixel_format=3)

    bridge._on_image(msg)

    ok, frame = bridge.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_bridge_read_frame_converts_protobuf_bgr():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    msg = _make_image_msg(width=160, height=120, channels=3, pixel_format=8)

    bridge._on_image(msg)

    ok, frame = bridge.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (120, 160, 3)


def test_bridge_read_frame_converts_greyscale():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    msg = _make_image_msg(width=100, height=80, channels=1, pixel_format=29)
    msg.step = 100

    bridge._on_image(msg)

    ok, frame = bridge.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (80, 100, 3)  # Greyscale expanded to BGR


def test_bridge_drops_old_frames():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)

    msg1 = _make_image_msg(width=100, height=100, channels=3, pixel_format=3)
    msg2 = _make_image_msg(width=200, height=150, channels=3, pixel_format=3)

    bridge._on_image(msg1)
    bridge._on_image(msg2)

    ok, frame = bridge.read_frame()
    assert ok is True
    assert frame.shape == (150, 200, 3)


def test_bridge_close_sets_disconnected():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic")
    bridge.open()
    assert bridge.health.connected
    bridge.close()
    assert not bridge.health.connected


def test_bridge_read_frame_returns_false_after_close():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    bridge.open()
    bridge.close()

    ok, frame = bridge.read_frame()
    assert ok is False
    assert frame is None


def test_bridge_write_health_json(tmp_path):
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic")
    bridge.open()
    health_path = tmp_path / "health.json"
    bridge.write_health_json(health_path)

    data = json.loads(health_path.read_text())
    assert data["source_type"] == "gazebo_camera"
    assert data["connected"] is True


def test_bridge_health_tracks_stale():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", target_fps=1000.0)
    msg = _make_image_msg(width=100, height=100, channels=3, pixel_format=3)
    bridge._on_image(msg)
    ok, frame = bridge.read_frame()
    assert ok is True

    time.sleep(0.05)
    h = bridge.health
    assert h.stale is True


def test_bridge_empty_image_message_ignored():
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    msg = _make_image_msg(width=0, height=0, channels=3, pixel_format=3)
    bridge._on_image(msg)

    ok, frame = bridge.read_frame()
    assert ok is False


def test_bridge_frame_timeout_on_subsequent_reads():
    """After first frame, subsequent read_frame waits frame_timeout_s then returns False."""
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=2.0, frame_timeout_s=0.2)
    bridge.open()
    msg = _make_image_msg(width=100, height=100, channels=3, pixel_format=3)
    bridge._on_image(msg)

    # First read gets the frame
    ok, frame = bridge.read_frame()
    assert ok is True

    # Second read waits frame_timeout_s then returns False (no frame, but connected)
    ok2, frame2 = bridge.read_frame()
    assert ok2 is False
    assert frame2 is None
    assert bridge.health.connected is True


def test_bridge_logs_first_image_metadata(capfd):
    """First image triggers a metadata log line."""
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    bridge = GazeboCameraBridge(topic="/test/topic", startup_timeout_s=0.2)
    msg = _make_image_msg(width=320, height=240, channels=3, pixel_format=3)
    bridge._on_image(msg)

    ok, frame = bridge.read_frame()
    assert ok is True
