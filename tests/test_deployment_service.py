"""Tests for Jetson service runner."""

from __future__ import annotations

from sentinel.common.event_bus import EventBus
from sentinel.config.schema import AppConfig
from sentinel.deployment.jetson_profile import PreflightReport
from sentinel.deployment.service import JetsonServiceRunner


def test_service_runner_creates() -> None:
    cfg = AppConfig()
    bus = EventBus()
    runner = JetsonServiceRunner(config=cfg, event_bus=bus)
    assert runner is not None


def test_run_preflight_returns_report() -> None:
    cfg = AppConfig()
    bus = EventBus()
    runner = JetsonServiceRunner(config=cfg, event_bus=bus)
    report = runner.run_preflight(check_camera=False, check_mavlink=False, allow_non_jetson=True)
    assert isinstance(report, PreflightReport)


def test_service_runner_setup_watchdog() -> None:
    cfg = AppConfig()
    bus = EventBus()
    runner = JetsonServiceRunner(config=cfg, event_bus=bus)
    wd = runner.setup_watchdog(mission_id="test_mission")
    assert wd is not None


def test_service_runner_setup_signal_handlers() -> None:
    cfg = AppConfig()
    bus = EventBus()
    runner = JetsonServiceRunner(config=cfg, event_bus=bus)
    runner.setup_signal_handlers()
    assert runner.shutdown_requested is False


def test_service_runner_stop_watchdog_twice() -> None:
    cfg = AppConfig()
    bus = EventBus()
    runner = JetsonServiceRunner(config=cfg, event_bus=bus)
    runner.setup_watchdog(mission_id="test")
    runner.stop_watchdog()
    runner.stop_watchdog()
