"""Tests for runtime watchdog."""

from __future__ import annotations

import time

from sentinel.common.event_bus import EventBus
from sentinel.common.events import SystemEvent
from sentinel.deployment.watchdog import RuntimeWatchdog


def test_watchdog_creates_and_stops() -> None:
    bus = EventBus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,
        sensor_stale_threshold_s=5.0,
        telemetry_stale_threshold_s=10.0,
        min_fps=0.5,
        ram_threshold_pct=99.0,
        disk_budget_mb=1,
        interval_s=0.2,
    )
    wd.start()
    time.sleep(0.4)
    wd.stop()
    assert wd.frame_count == 0


def test_watchdog_note_frame_received() -> None:
    bus = EventBus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,
        sensor_stale_threshold_s=1.0,
        telemetry_stale_threshold_s=10.0,
        min_fps=0.5,
        ram_threshold_pct=99.0,
        disk_budget_mb=1,
        interval_s=0.2,
    )
    wd.start()
    wd.note_frame_received()
    time.sleep(0.4)
    wd.stop()


def test_watchdog_note_telemetry_received() -> None:
    bus = EventBus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,
        sensor_stale_threshold_s=1.0,
        telemetry_stale_threshold_s=10.0,
        min_fps=0.5,
        ram_threshold_pct=99.0,
        disk_budget_mb=1,
        interval_s=0.2,
    )
    wd.start()
    wd.note_telemetry_received()
    time.sleep(0.4)
    wd.stop()


def test_watchdog_fires_sensor_stale_after_timeout() -> None:
    collected: list[str] = []

    class _Bus:
        def publish(self, event: SystemEvent) -> None:
            collected.append(event.event_type)

        def subscribe(self, *args, **kwargs) -> None:
            pass

        def clear(self) -> None:
            collected.clear()

    bus = _Bus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,  # type: ignore[arg-type]
        sensor_stale_threshold_s=0.15,
        telemetry_stale_threshold_s=10.0,
        min_fps=0.1,
        ram_threshold_pct=99.0,
        disk_budget_mb=1,
        interval_s=0.1,
    )
    wd.note_frame_received()
    wd.start()
    time.sleep(0.8)
    wd.stop()

    assert "watchdog_sensor_stale" in collected, f"Got events: {collected}"


def test_watchdog_fires_telemetry_stale_after_timeout() -> None:
    collected: list[str] = []

    class _Bus:
        def publish(self, event: SystemEvent) -> None:
            collected.append(event.event_type)

        def subscribe(self, *args, **kwargs) -> None:
            pass

        def clear(self) -> None:
            collected.clear()

    bus = _Bus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,  # type: ignore[arg-type]
        sensor_stale_threshold_s=10.0,
        telemetry_stale_threshold_s=0.15,
        min_fps=0.1,
        ram_threshold_pct=99.0,
        disk_budget_mb=1,
        interval_s=0.1,
    )
    wd.note_telemetry_received()
    wd.start()
    time.sleep(0.8)
    wd.stop()

    assert "watchdog_telemetry_stale" in collected, f"Got events: {collected}"


def test_watchdog_sensor_recovery_after_frame() -> None:
    collected: list[str] = []

    class _Bus:
        def publish(self, event: SystemEvent) -> None:
            collected.append(event.event_type)

        def subscribe(self, *args, **kwargs) -> None:
            pass

        def clear(self) -> None:
            collected.clear()

    bus = _Bus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,  # type: ignore[arg-type]
        sensor_stale_threshold_s=0.15,
        telemetry_stale_threshold_s=10.0,
        min_fps=0.1,
        ram_threshold_pct=99.0,
        disk_budget_mb=1,
        interval_s=0.1,
    )
    wd.note_frame_received()
    wd.start()
    time.sleep(0.5)
    wd.note_frame_received()
    time.sleep(0.3)
    wd.stop()

    assert "watchdog_sensor_recovered" in collected, f"Got events: {collected}"


def test_watchdog_double_start_is_safe() -> None:
    bus = EventBus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,
        interval_s=0.2,
    )
    wd.start()
    wd.start()
    time.sleep(0.3)
    wd.stop()


def test_watchdog_multiple_frames_updates_count() -> None:
    bus = EventBus()
    wd = RuntimeWatchdog(
        mission_id="test",
        event_bus=bus,
        interval_s=0.2,
    )
    for _ in range(5):
        wd.note_frame_received()
    assert wd.frame_count == 5
