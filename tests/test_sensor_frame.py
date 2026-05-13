"""Tests for the SensorFrame model and timestamp discipline."""

from __future__ import annotations

import time

from sentinel.common.types import SensorFrame


def test_sensor_frame_creation() -> None:
    now_mono = time.monotonic()
    sf = SensorFrame(
        frame_id=1,
        capture_timestamp_monotonic=now_mono,
        capture_timestamp_utc="2026-01-01T00:00:00.000000Z",
        receive_timestamp_monotonic=now_mono + 0.05,
        source="test_camera",
    )
    assert sf.frame_id == 1
    assert sf.capture_timestamp_monotonic == now_mono
    assert sf.receive_timestamp_monotonic >= sf.capture_timestamp_monotonic


def test_sensor_frame_delta_positive() -> None:
    t0 = time.monotonic()
    sf = SensorFrame(
        frame_id=0,
        capture_timestamp_monotonic=t0,
        capture_timestamp_utc="2026-01-01T00:00:00.000000Z",
        receive_timestamp_monotonic=t0 + 0.033,
        source="test",
    )
    delta_ms = (sf.receive_timestamp_monotonic - sf.capture_timestamp_monotonic) * 1000.0
    assert delta_ms > 0


def test_sensor_frame_metadata_defaults() -> None:
    sf = SensorFrame(
        frame_id=0,
        capture_timestamp_monotonic=1.0,
        capture_timestamp_utc="x",
        receive_timestamp_monotonic=1.0,
    )
    assert sf.metadata == {}
    assert sf.source == "unknown"


def test_sensor_frame_serializes() -> None:
    sf = SensorFrame(
        frame_id=5,
        capture_timestamp_monotonic=100.0,
        capture_timestamp_utc="2026-01-01T00:00:00.000000Z",
        receive_timestamp_monotonic=100.05,
        source="webcam:0",
        metadata={"width": 640, "height": 480},
    )
    d = sf.model_dump()
    assert d["frame_id"] == 5
    assert d["source"] == "webcam:0"
    assert d["metadata"]["width"] == 640
