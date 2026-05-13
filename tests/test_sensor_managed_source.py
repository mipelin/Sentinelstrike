"""Tests for managed video source and factory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sentinel.common.event_bus import EventBus
from sentinel.config.schema import VideoSourceConfig
from sentinel.sensors.factory import create_video_source
from sentinel.sensors.managed_source import ManagedVideoSource
from sentinel.sensors.types import VideoSourceType


def _create_video(path: Path, frames: int = 5, w: int = 160, h: int = 120) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (w, h))
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (i * 50 % 255, 128, 128)
        writer.write(frame)
    writer.release()
    return path


# --- File source tests ---

def test_file_source_reads_frames(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=5)
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
    )
    assert src.open() is True
    count = 0
    while True:
        ret, frame = src.read_frame()
        if not ret:
            break
        assert frame is not None
        count += 1
    src.close()
    assert count == 5


def test_file_source_health_connected(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=3)
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
    )
    src.open()
    assert src.health.connected is True
    src.close()
    assert src.health.connected is False


def test_file_source_missing_returns_false(tmp_path: Path):
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    assert src.open() is False


def test_file_source_iterator(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=4)
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
    )
    src.open()
    frames = list(src)
    src.close()
    assert len(frames) == 4
    assert all("frame" in f for f in frames)


def test_file_source_fps_estimate(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=10)
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
    )
    src.open()
    for _ in range(10):
        src.read_frame()
    src.close()
    # FPS should be > 0 for file reads (they're fast)
    assert src.health.fps_estimate > 0


# --- Webcam tests (mocked) ---

@patch("sentinel.sensors.managed_source.cv2.VideoCapture")
def test_webcam_missing_returns_false(mock_vcap_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vcap_cls.return_value = mock_cap

    src = ManagedVideoSource(
        source_type=VideoSourceType.WEBCAM,
        source_label="webcam:0",
        webcam_index=0,
    )
    assert src.open() is False


@patch("sentinel.sensors.managed_source.cv2.VideoCapture")
def test_webcam_sets_resolution(mock_vcap_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_vcap_cls.return_value = mock_cap

    src = ManagedVideoSource(
        source_type=VideoSourceType.WEBCAM,
        source_label="webcam:0",
        webcam_index=0,
        frame_width=1280,
        frame_height=720,
    )
    assert src.open() is True
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)


@patch("sentinel.sensors.managed_source.cv2.VideoCapture")
def test_webcam_reads_frames(mock_vcap_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, frame),
        (True, frame),
        (False, None),
    ]
    mock_vcap_cls.return_value = mock_cap

    src = ManagedVideoSource(
        source_type=VideoSourceType.WEBCAM,
        source_label="webcam:0",
        webcam_index=0,
        reconnect_enabled=False,
    )
    src.open()
    ret1, f1 = src.read_frame()
    assert ret1 and f1 is not None
    ret2, f2 = src.read_frame()
    assert ret2 and f2 is not None
    ret3, f3 = src.read_frame()
    assert ret3 is False
    src.close()


# --- RTSP tests (mocked) ---

@patch("sentinel.sensors.managed_source.time.sleep")
@patch("sentinel.sensors.managed_source.cv2.VideoCapture")
def test_rtsp_reconnects_on_drop(mock_vcap_cls, mock_sleep):
    call_count = 0

    def make_cap(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        mock.isOpened.return_value = True
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if call_count == 1:
            mock.read.side_effect = [
                (True, frame),
                (False, None),  # drop
            ]
        else:
            mock.read.return_value = (True, frame)
        return mock

    mock_vcap_cls.side_effect = make_cap

    event_bus = EventBus()
    events: list = []
    event_bus.subscribe("*", lambda e: events.append(e))

    src = ManagedVideoSource(
        source_type=VideoSourceType.RTSP,
        source_label="rtsp://test",
        rtsp_url="rtsp://test",
        reconnect_enabled=True,
        reconnect_interval_s=1.0,
        event_bus=event_bus,
        mission_id="test",
    )
    src.open()

    ret1, _ = src.read_frame()
    assert ret1 is True

    ret2, _ = src.read_frame()
    assert ret2 is True  # reconnect succeeds

    event_types = [e.event_type for e in events]
    assert "sensor_connected" in event_types
    assert "sensor_disconnected" in event_types
    assert "sensor_reconnected" in event_types

    assert src.health.reconnect_count >= 1
    src.close()


@patch("sentinel.sensors.managed_source.cv2.VideoCapture")
def test_rtsp_missing_returns_false(mock_vcap_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vcap_cls.return_value = mock_cap

    src = ManagedVideoSource(
        source_type=VideoSourceType.RTSP,
        source_label="rtsp://nothing",
        rtsp_url="rtsp://nothing",
    )
    assert src.open() is False


# --- Health tests ---

def test_write_health_json(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=3)
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
    )
    src.open()
    for _ in range(3):
        src.read_frame()

    health_path = tmp_path / "sensor_health.json"
    src.close()
    src.write_health_json(health_path)

    data = json.loads(health_path.read_text())
    assert data["connected"] is False  # closed
    assert data["dropped_frames"] == 0
    assert data["reconnect_count"] == 0
    assert data["source_type"] == "file"
    assert data["last_frame_utc"] is not None


def test_health_stale_detection(tmp_path: Path):
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="test",
        target_fps=100.0,
    )
    # No frames read, last_frame_monotonic is 0 → not stale (no frame yet)
    assert src.health.stale is False


def test_dropped_frames_count(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=3)
    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
    )
    src.open()
    for _ in range(3):
        src.read_frame()
    # File source returns False on EOF, doesn't count as dropped
    src.read_frame()
    assert src.health.dropped_frames == 0
    src.close()


# --- Event publishing tests ---

def test_file_source_publishes_connected_event(tmp_path: Path):
    video = _create_video(tmp_path / "test.mp4", frames=2)
    event_bus = EventBus()
    events: list = []
    event_bus.subscribe("*", lambda e: events.append(e))

    src = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label=str(video),
        file_path=str(video),
        event_bus=event_bus,
        mission_id="test",
    )
    src.open()
    src.close()

    types = [e.event_type for e in events]
    assert "sensor_connected" in types


# --- Factory tests ---

def test_factory_creates_file_source():
    cfg = VideoSourceConfig(source_type="file", file_path="test.mp4")
    src = create_video_source(cfg)
    assert src._source_type == VideoSourceType.FILE
    assert src._file_path == "test.mp4"


def test_factory_creates_webcam_source():
    cfg = VideoSourceConfig(source_type="webcam", webcam_index=1)
    src = create_video_source(cfg)
    assert src._source_type == VideoSourceType.WEBCAM
    assert src._webcam_index == 1


def test_factory_creates_rtsp_source():
    cfg = VideoSourceConfig(source_type="rtsp", rtsp_url="rtsp://example.com/stream")
    src = create_video_source(cfg)
    assert src._source_type == VideoSourceType.RTSP
    assert src._rtsp_url == "rtsp://example.com/stream"


def test_factory_rtsp_without_url_raises():
    cfg = VideoSourceConfig(source_type="rtsp", rtsp_url="")
    with pytest.raises(ValueError, match="rtsp_url is required"):
        create_video_source(cfg)
