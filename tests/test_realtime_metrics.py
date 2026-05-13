"""Tests for realtime metrics computation."""

from sentinel.realtime.metrics import compute_realtime_metrics


def test_empty_loop_times():
    m = compute_realtime_metrics(
        frame_count=0,
        detection_count=0,
        track_count=0,
        geo_observation_count=0,
        tak_message_count=0,
        loop_times_ms=[],
        telemetry_stale_count=0,
        started_at_utc="2026-01-01T00:00:00.000000Z",
        ended_at_utc="2026-01-01T00:00:01.000000Z",
        wall_clock_s=1.0,
        target_fps=10.0,
    )
    assert m["average_loop_hz"] == 0.0
    assert m["average_frame_processing_ms"] == 0.0
    assert m["frame_count"] == 0
    assert m["achieved_fps"] == 0.0
    assert m["target_fps"] == 10.0


def test_with_loop_times_and_wall_clock():
    m = compute_realtime_metrics(
        frame_count=10,
        detection_count=5,
        track_count=3,
        geo_observation_count=2,
        tak_message_count=4,
        loop_times_ms=[5.0, 5.0, 5.0],
        telemetry_stale_count=0,
        started_at_utc="2026-01-01T00:00:00.000000Z",
        ended_at_utc="2026-01-01T00:01:00.000000Z",
        wall_clock_s=1.0,
        target_fps=10.0,
    )
    assert m["frame_count"] == 10
    assert m["average_loop_hz"] == 10.0
    assert m["achieved_fps"] == 10.0
    assert m["average_frame_processing_ms"] == 5.0
    assert m["target_fps"] == 10.0
    assert m["wall_clock_duration_s"] == 1.0
    assert abs(m["average_frame_period_ms"] - 100.0) < 0.01


def test_stale_count_preserved():
    m = compute_realtime_metrics(
        frame_count=1,
        detection_count=0,
        track_count=0,
        geo_observation_count=0,
        tak_message_count=0,
        loop_times_ms=[100.0],
        telemetry_stale_count=3,
        started_at_utc="s",
        ended_at_utc="e",
        wall_clock_s=0.5,
        target_fps=2.0,
    )
    assert m["telemetry_stale_count"] == 3
    assert m["achieved_fps"] == 2.0


def test_timestamps_preserved():
    m = compute_realtime_metrics(
        frame_count=1,
        detection_count=0,
        track_count=0,
        geo_observation_count=0,
        tak_message_count=0,
        loop_times_ms=[50.0],
        telemetry_stale_count=0,
        started_at_utc="2026-05-11T12:00:00.000000Z",
        ended_at_utc="2026-05-11T12:00:01.000000Z",
        wall_clock_s=1.0,
        target_fps=5.0,
    )
    assert m["started_at_utc"] == "2026-05-11T12:00:00.000000Z"
    assert m["ended_at_utc"] == "2026-05-11T12:00:01.000000Z"


def test_achieved_fps_from_wall_clock():
    m = compute_realtime_metrics(
        frame_count=30,
        detection_count=0,
        track_count=0,
        geo_observation_count=0,
        tak_message_count=0,
        loop_times_ms=[1.0] * 30,
        telemetry_stale_count=0,
        started_at_utc="s",
        ended_at_utc="e",
        wall_clock_s=3.0,
        target_fps=10.0,
    )
    assert m["achieved_fps"] == 10.0
    assert m["average_loop_hz"] == 10.0
    # Processing time is 1ms but achieved FPS is 10 due to wall clock
    assert m["average_frame_processing_ms"] == 1.0


def test_zero_wall_clock():
    m = compute_realtime_metrics(
        frame_count=5,
        detection_count=0,
        track_count=0,
        geo_observation_count=0,
        tak_message_count=0,
        loop_times_ms=[1.0],
        telemetry_stale_count=0,
        started_at_utc="s",
        ended_at_utc="e",
        wall_clock_s=0.0,
        target_fps=10.0,
    )
    assert m["achieved_fps"] == 0.0
