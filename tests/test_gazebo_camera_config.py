"""Tests for Gazebo camera source type config and factory integration."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from sentinel.config.schema import VideoSourceConfig
from sentinel.sensors.types import VideoSourceType


def test_video_source_type_enum_has_gazebo_camera():
    assert VideoSourceType.GAZEBO_CAMERA == "gazebo_camera"


def test_config_accepts_gazebo_camera_source_type():
    cfg = VideoSourceConfig(
        source_type="gazebo_camera",
        gazebo_camera_topic="/world/default/model/x500_mono_cam/link/camera_link/sensor/camera/image",
    )
    assert cfg.source_type == "gazebo_camera"


def test_config_rejects_invalid_source_type():
    with pytest.raises(Exception):
        VideoSourceConfig(source_type="invalid_type")


def test_config_gazebo_fields_defaults():
    cfg = VideoSourceConfig(source_type="gazebo_camera")
    assert cfg.gazebo_camera_topic == ""
    assert cfg.gazebo_world_name == "default"


def test_factory_requires_topic_for_gazebo_camera():
    from sentinel.sensors.factory import create_video_source

    cfg = VideoSourceConfig(
        source_type="gazebo_camera",
        gazebo_camera_topic="",
    )
    with pytest.raises(ValueError, match="gazebo_camera_topic is required"):
        create_video_source(cfg)


def _setup_gz_mocks(monkeypatch):
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
    if "sentinel.sensors.gazebo_camera_bridge" in sys.modules:
        del sys.modules["sentinel.sensors.gazebo_camera_bridge"]


def test_factory_creates_gazebo_camera_bridge(monkeypatch):
    _setup_gz_mocks(monkeypatch)

    from sentinel.sensors.factory import create_video_source
    from sentinel.sensors.gazebo_camera_bridge import GazeboCameraBridge

    cfg = VideoSourceConfig(
        source_type="gazebo_camera",
        gazebo_camera_topic="/test/camera/image",
    )
    provider = create_video_source(cfg)
    assert isinstance(provider, GazeboCameraBridge)
    assert provider._topic == "/test/camera/image"


def test_factory_creates_managed_source_for_rtsp():
    from sentinel.sensors.factory import create_video_source
    from sentinel.sensors.managed_source import ManagedVideoSource

    cfg = VideoSourceConfig(
        source_type="rtsp",
        rtsp_url="rtsp://localhost:8554/test",
    )
    provider = create_video_source(cfg)
    assert isinstance(provider, ManagedVideoSource)


def test_factory_creates_managed_source_for_file():
    from sentinel.sensors.factory import create_video_source
    from sentinel.sensors.managed_source import ManagedVideoSource

    cfg = VideoSourceConfig(source_type="file", file_path="test.mp4")
    provider = create_video_source(cfg)
    assert isinstance(provider, ManagedVideoSource)
