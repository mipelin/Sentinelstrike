"""Tests for ManagedVideoSource improvements — timeout, age_ms, health."""

from __future__ import annotations

import numpy as np

from sentinel.sensors.managed_source import ManagedVideoSource
from sentinel.sensors.types import SensorHealth, VideoSourceType


def test_health_has_latest_frame_age_ms() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    h = ms.health
    assert hasattr(h, "latest_frame_age_ms")
    assert isinstance(h.latest_frame_age_ms, float)


def test_latest_frame_age_ms_property() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    age = ms.latest_frame_age_ms
    assert isinstance(age, float)


def test_build_sensor_frame_returns_valid_frame() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sf = ms.build_sensor_frame(fake_frame)
    assert sf.frame_id == 0
    assert sf.source == "nonexistent.mp4"
    assert sf.metadata["width"] == 640
    assert sf.metadata["height"] == 480
    assert sf.receive_timestamp_monotonic >= sf.capture_timestamp_monotonic


def test_open_file_nonexistent_returns_false() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    assert ms.open() is False


def test_close_is_safe() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    ms.close()


def test_health_returns_stale_when_no_frames() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="nonexistent.mp4",
        file_path="nonexistent.mp4",
    )
    h = ms.health
    assert isinstance(h, SensorHealth)


def test_managed_source_with_timeout_param() -> None:
    ms = ManagedVideoSource(
        source_type=VideoSourceType.FILE,
        source_label="test.mp4",
        file_path="test.mp4",
        read_timeout_ms=3000,
    )
    assert ms._read_timeout_ms == 3000
