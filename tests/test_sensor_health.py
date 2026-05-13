"""Tests for sensor health dataclass."""


from sentinel.sensors.types import SensorHealth, VideoSourceType


def test_sensor_health_defaults():
    h = SensorHealth()
    assert h.connected is False
    assert h.fps_estimate == 0.0
    assert h.dropped_frames == 0
    assert h.reconnect_count == 0
    assert h.last_frame_utc is None
    assert h.stale is False
    assert h.source_type == VideoSourceType.FILE
    assert h.source_label == ""


def test_sensor_health_with_values():
    h = SensorHealth(
        connected=True,
        fps_estimate=15.3,
        dropped_frames=2,
        reconnect_count=1,
        last_frame_utc="2026-05-11T12:00:00Z",
        source_type=VideoSourceType.RTSP,
        source_label="rtsp://test",
    )
    assert h.connected is True
    assert h.fps_estimate == 15.3
    assert h.dropped_frames == 2
    assert h.source_type == VideoSourceType.RTSP


def test_sensor_health_mutable():
    h = SensorHealth()
    h.connected = True
    h.dropped_frames += 1
    assert h.connected is True
    assert h.dropped_frames == 1
