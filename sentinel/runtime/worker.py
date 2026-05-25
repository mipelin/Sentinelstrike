"""Worker base class for loop-based runtime services."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from sentinel.runtime.metrics import WorkerMetrics


@dataclass
class WorkerHealth:
    """Health snapshot for a single Worker."""

    name: str
    running: bool
    stale: bool
    metrics: dict  # from WorkerMetrics.snapshot()


class Worker(ABC):
    """Base class for loop-based runtime workers.

    Provides: start/stop lifecycle, tick-based loop, exception isolation,
    Hz metrics, and health reporting.

    Subclasses override tick() to perform one iteration of work.
    tick() must not raise unhandled exceptions -- any exception is caught,
    logged, and counted in metrics.

    Usage::

        class MyWorker(Worker):
            def tick(self) -> None:
                ...

        w = MyWorker(name="my_worker", hz=10.0)
        w.start()
        health = w.health()
        w.stop()
    """

    def __init__(self, name: str, hz: float = 10.0) -> None:
        self._name = name
        self._hz = hz
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._metrics = WorkerMetrics(name=name)
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._running

    @property
    def metrics(self) -> WorkerMetrics:
        return self._metrics

    def start(self) -> None:
        """Start the worker loop in a daemon thread.

        No-op if already running (thread is alive).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=self._name,
        )
        self._thread.start()
        self._running = True
        logger.info("Worker {} started (target {:.1f} Hz)", self._name, self._hz)

    def stop(self) -> None:
        """Signal stop and join the worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._running = False
        logger.info("Worker {} stopped", self._name)

    def health(self) -> WorkerHealth:
        """Return current health snapshot."""
        stale_threshold_s = 3.0 / max(self._hz, 1.0)
        return WorkerHealth(
            name=self._name,
            running=self._running,
            stale=self._metrics.last_tick_age_s() > stale_threshold_s,
            metrics=self._metrics.snapshot(),
        )

    @abstractmethod
    def tick(self) -> None:
        """Override: one iteration of work.

        Must not raise -- exceptions are caught by the _run loop.
        """
        ...

    def _run(self) -> None:
        """Main loop with exception isolation and timing."""
        interval = 1.0 / self._hz if self._hz > 0 else 0.1
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception:
                self._metrics._exceptions += 1
                logger.exception("Worker {} tick failed", self._name)
            elapsed = time.monotonic() - t0
            self._metrics.record_tick(elapsed_s=elapsed)
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
