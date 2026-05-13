"""Jetson service runner — wraps the agent with watchdog, logging, and signal handling."""

from __future__ import annotations

import signal
from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.config.schema import AppConfig
from sentinel.deployment.jetson_profile import PreflightReport, run_preflight
from sentinel.deployment.watchdog import RuntimeWatchdog


class JetsonServiceRunner:
    """Production service runner for Jetson companion computer deployment.

    Handles:
    - Preflight resource checks
    - Signal handling (SIGINT/SIGTERM for graceful shutdown)
    - Watchdog lifecycle
    - Safe shutdown sequence
    """

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self._watchdog: RuntimeWatchdog | None = None
        self._shutdown_flag = False

    def run_preflight(
        self,
        *,
        check_camera: bool = False,
        check_mavlink: bool = False,
        strict: bool = True,
        allow_non_jetson: bool = False,
        skip_camera: bool = False,
        skip_mavlink: bool = False,
    ) -> PreflightReport:
        cfg = self._config
        return run_preflight(
            model_path=cfg.perception.model_path,
            source_type=cfg.video.source_type,
            rtsp_url=cfg.video.rtsp_url,
            webcam_index=cfg.video.webcam_index,
            file_path=cfg.video.file_path,
            mavlink_connection_url=cfg.mavlink.connection_url if check_mavlink else "",
            disk_budget_mb=cfg.runtime.disk_budget_mb,
            min_memory_mb=512,
            check_mavlink=check_mavlink,
            check_camera=check_camera,
            strict=strict,
            allow_non_jetson=allow_non_jetson,
            skip_camera=skip_camera,
            skip_mavlink=skip_mavlink,
        )

    def setup_watchdog(self, mission_id: str) -> RuntimeWatchdog:
        cfg = self._config.runtime
        wd = RuntimeWatchdog(
            mission_id=mission_id,
            event_bus=self._event_bus,
            sensor_stale_threshold_s=cfg.watchdog_sensor_stale_s,
            telemetry_stale_threshold_s=cfg.watchdog_telemetry_stale_s,
            min_fps=cfg.watchdog_min_fps,
            ram_threshold_pct=cfg.watchdog_ram_threshold_pct,
            disk_budget_mb=cfg.disk_budget_mb,
            disk_path=str(Path(self._config.recorder.output_dir)),
        )
        self._watchdog = wd
        return wd

    def setup_signal_handlers(self) -> None:
        def _handler(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received signal {} — initiating graceful shutdown", sig_name)
            self._shutdown_flag = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_flag

    def stop_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
