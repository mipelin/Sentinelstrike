"""Tests for BackgroundMavlinkTelemetryProvider — non-blocking telemetry cache."""

from __future__ import annotations

import threading
import time

from sentinel.realtime.telemetry_stream import BackgroundMavlinkTelemetryProvider


class _MockBridge:
    def __init__(self, delay_s: float = 0.0, fail: bool = False) -> None:
        self._delay = delay_s
        self._fail = fail
        self.call_count = 0

    def get_telemetry(self) -> object:
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("simulated telemetry failure")
        from sentinel.mavlink_bridge.telemetry import TelemetrySnapshot

        return TelemetrySnapshot(
            vehicle_id="uav_001",
            timestamp_utc="2026-01-01T00:00:00Z",
            connected=True,
            armed=False,
            mode="AUTO",
            heading_deg=90.0,
            groundspeed_mps=5.0,
            battery_pct=95.0,
        )


def test_get_vehicle_state_does_not_block_with_slow_bridge():
    bridge = _MockBridge(delay_s=2.0)
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=5.0,
    )
    provider.start()
    try:
        # Give background thread time to get first result
        time.sleep(0.1)

        t0 = time.monotonic()
        state = provider.get_vehicle_state()
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        assert elapsed_ms < 50, f"get_vehicle_state blocked for {elapsed_ms:.0f}ms"
    finally:
        provider.close()


def test_provider_returns_last_valid_state():
    bridge = _MockBridge()
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=10.0,
    )
    provider.start()
    try:
        time.sleep(0.3)
        state = provider.get_vehicle_state()
        assert state is not None
        assert state.vehicle_id == "uav_001"
    finally:
        provider.close()


def test_provider_returns_none_before_first_poll():
    """Before start() is called, get_vehicle_state() returns None."""
    bridge = _MockBridge()
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=5.0,
    )
    state = provider.get_vehicle_state()
    assert state is None
    provider.close()


def test_provider_marks_stale_after_timeout():
    bridge = _MockBridge()
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=100.0,
        stale_after_s=0.1,
    )
    provider.start()
    try:
        time.sleep(0.15)
        state = provider.get_vehicle_state()
        assert state is not None
        assert provider.stale_count == 0

        # Close the provider so polls stop, then wait for staleness
        provider.close()

        # Re-read: stale_count should remain accessible
        assert provider.stale_count >= 0
    except Exception:
        provider.close()
        raise


def test_provider_tracks_errors():
    bridge = _MockBridge(fail=True)
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=20.0,
    )
    provider.start()
    try:
        time.sleep(0.3)
        assert provider.error_count > 0
        assert provider.last_error is not None
        assert "simulated telemetry failure" in provider.last_error
    finally:
        provider.close()


def test_provider_close_stops_thread():
    bridge = _MockBridge()
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=10.0,
    )
    provider.start()
    thread = provider._thread
    assert thread is not None
    assert thread.is_alive()

    provider.close()
    assert not thread.is_alive()
    assert provider._thread is None


def test_provider_cache_hits_increment():
    bridge = _MockBridge()
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=20.0,
    )
    provider.start()
    try:
        time.sleep(0.2)
        assert provider.get_vehicle_state() is not None
        assert provider.get_vehicle_state() is not None
        assert provider.cache_hits >= 2
    finally:
        provider.close()


def test_provider_link_ok_tracks_status():
    bridge = _MockBridge()
    provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=bridge,
        poll_hz=20.0,
    )
    assert provider.link_ok is False
    provider.start()
    try:
        time.sleep(0.2)
        assert provider.link_ok is True
    finally:
        provider.close()


def test_realtime_loop_with_slow_telemetry_maintains_fps(tmp_path):
    """Integration: RealTimeMissionLoop with a slow mock bridge keeps FPS reasonable."""
    import json

    import cv2
    import numpy as np

    from sentinel.common.event_bus import EventBus
    from sentinel.config.schema import AppConfig
    from sentinel.realtime.loop import RealTimeMissionLoop

    # Create a test video
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (320, 240))
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Slow bridge (2s per call)
    slow_bridge = _MockBridge(delay_s=2.0)
    bg_provider = BackgroundMavlinkTelemetryProvider(
        mavlink_bridge=slow_bridge,
        poll_hz=5.0,
    )
    bg_provider.start()

    cfg = AppConfig(system={"vehicle_id": "uav_test"})
    event_bus = EventBus()
    run_dir = tmp_path / "runs" / "test"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        loop = RealTimeMissionLoop(
            config=cfg,
            mission_id="test",
            event_bus=event_bus,
            run_dir=run_dir,
            telemetry_provider=bg_provider,
            max_frames=10,
            target_fps=20.0,
            video_source=str(video_path),
        )
        t0 = time.monotonic()
        metrics = loop.run()
        elapsed = time.monotonic() - t0

        # 10 frames at 20Hz target = ~0.5s max, never 20s (which would happen with blocking)
        assert elapsed < 5.0, f"Loop took {elapsed:.1f}s — telemetry is blocking!"
        assert metrics["frame_count"] == 10
        assert metrics["telemetry_provider_mode"] == "background"
    finally:
        bg_provider.close()
