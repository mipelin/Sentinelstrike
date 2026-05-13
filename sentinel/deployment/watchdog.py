"""Runtime watchdog — monitors sensor, telemetry, loop health, disk, and RAM.

Publishes watchdog events via EventBus. Does NOT execute safety actions directly —
the existing SafetyActionExecutor reads events and decides.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.deployment.resources import disk_budget_exceeded, ram_pressure_pct


@dataclass
class WatchdogState:
    last_frame_monotonic: float = 0.0
    last_telemetry_monotonic: float = 0.0
    sensor_stale_fired: bool = False
    telemetry_stale_fired: bool = False
    low_fps_fired: bool = False
    disk_low_fired: bool = False
    ram_high_fired: bool = False
    warning_fired: set[str | None] = None

    def __post_init__(self) -> None:
        if self.warning_fired is None:
            self.warning_fired = set()


class RuntimeWatchdog:
    """Monitors runtime health and publishes events on degradation.

    Runs as a background thread that samples state every `interval_s` seconds.
    """

    def __init__(
        self,
        mission_id: str,
        event_bus: EventBus,
        *,
        sensor_stale_threshold_s: float = 5.0,
        telemetry_stale_threshold_s: float = 10.0,
        min_fps: float = 2.0,
        ram_threshold_pct: float = 90.0,
        disk_budget_mb: int = 2048,
        disk_path: str = ".",
        interval_s: float = 2.0,
    ) -> None:
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._sensor_stale_s = sensor_stale_threshold_s
        self._telemetry_stale_s = telemetry_stale_threshold_s
        self._min_fps = min_fps
        self._ram_threshold_pct = ram_threshold_pct
        self._disk_budget_mb = disk_budget_mb
        self._disk_path = disk_path
        self._interval_s = interval_s

        self._state = WatchdogState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_count: int = 0
        self._fps_samples: list[tuple[float, int]] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="runtime-watchdog")
        self._thread.start()
        logger.info("Runtime watchdog started (interval={:.1f}s)", self._interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Runtime watchdog stopped")

    def note_frame_received(self) -> None:
        self._state.last_frame_monotonic = time.monotonic()
        self._frame_count += 1
        if self._state.sensor_stale_fired:
            self._state.sensor_stale_fired = False
            self._publish("watchdog_sensor_recovered", {"reason": "frame received"})

    def note_telemetry_received(self) -> None:
        self._state.last_telemetry_monotonic = time.monotonic()
        if self._state.telemetry_stale_fired:
            self._state.telemetry_stale_fired = False
            self._publish("watchdog_telemetry_recovered", {"reason": "telemetry received"})

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _publish(self, event_type: str, payload: dict | None = None) -> None:
        try:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="runtime_watchdog",
                    event_type=event_type,
                    severity="warning",
                    payload=payload or {},
                )
            )
        except Exception:
            logger.exception("Watchdog event publish failed")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample()
            except Exception:
                logger.exception("Watchdog sample error")
            self._stop_event.wait(self._interval_s)

    def _sample(self) -> None:
        now = time.monotonic()

        # Sensor staleness
        if self._state.last_frame_monotonic > 0:
            frame_age = now - self._state.last_frame_monotonic
            if frame_age > self._sensor_stale_s and not self._state.sensor_stale_fired:
                self._state.sensor_stale_fired = True
                self._publish("watchdog_sensor_stale", {"age_s": round(frame_age, 2)})
                logger.warning("Watchdog: sensor stale for {:.1f}s", frame_age)

        # Telemetry staleness
        if self._state.last_telemetry_monotonic > 0:
            telem_age = now - self._state.last_telemetry_monotonic
            if telem_age > self._telemetry_stale_s and not self._state.telemetry_stale_fired:
                self._state.telemetry_stale_fired = True
                self._publish("watchdog_telemetry_stale", {"age_s": round(telem_age, 2)})
                logger.warning("Watchdog: telemetry stale for {:.1f}s", telem_age)

        # FPS tracking
        self._fps_samples.append((now, self._frame_count))
        cutoff = now - 10.0
        self._fps_samples = [(t, c) for t, c in self._fps_samples if t > cutoff]
        if len(self._fps_samples) >= 2:
            dt = self._fps_samples[-1][0] - self._fps_samples[0][0]
            df = self._fps_samples[-1][1] - self._fps_samples[0][1]
            if dt > 0:
                fps = df / dt
                if fps < self._min_fps and not self._state.low_fps_fired:
                    self._state.low_fps_fired = True
                    self._publish("watchdog_low_fps", {"fps": round(fps, 2), "min": self._min_fps})
                    logger.warning("Watchdog: low FPS {:.1f} (min {:.1f})", fps, self._min_fps)
                elif fps >= self._min_fps and self._state.low_fps_fired:
                    self._state.low_fps_fired = False
                    self._publish("watchdog_fps_recovered", {"fps": round(fps, 2)})

        # Disk check
        if disk_budget_exceeded(self._disk_path, self._disk_budget_mb):
            if not self._state.disk_low_fired:
                self._state.disk_low_fired = True
                self._publish("watchdog_disk_low", {"budget_mb": self._disk_budget_mb})
                logger.warning("Watchdog: disk free space below {} MB budget", self._disk_budget_mb)
        elif self._state.disk_low_fired:
            self._state.disk_low_fired = False
            self._publish("watchdog_disk_recovered")

        # RAM check
        ram_pct = ram_pressure_pct()
        if ram_pct > self._ram_threshold_pct and not self._state.ram_high_fired:
            self._state.ram_high_fired = True
            self._publish("watchdog_ram_high", {"percent_used": round(ram_pct, 1)})
            logger.warning("Watchdog: RAM usage {:.1f}% exceeds threshold {:.1f}%", ram_pct, self._ram_threshold_pct)
        elif ram_pct <= self._ram_threshold_pct and self._state.ram_high_fired:
            self._state.ram_high_fired = False
            self._publish("watchdog_ram_recovered")
