"""Per-worker timing and error statistics."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field


@dataclass
class WorkerMetrics:
    """Rolling-window timing and error counters for a single Worker.

    Not thread-safe by itself. Intended to be mutated from the worker's
    own thread and snapshotted for external consumption.
    """

    name: str
    _tick_times: list[float] = field(default_factory=list)
    _wall_times: list[float] = field(default_factory=list)
    _tick_count: int = 0
    _skip_count: int = 0
    _exceptions: int = 0
    _stale_reads: int = 0
    _last_tick_monotonic: float = 0.0
    _window_size: int = 100

    def record_tick(self, elapsed_s: float) -> None:
        """Record one tick duration in seconds."""
        now = time.monotonic()
        self._tick_times.append(elapsed_s)
        self._wall_times.append(now)
        if len(self._tick_times) > self._window_size:
            self._tick_times = self._tick_times[-self._window_size :]
        if len(self._wall_times) > self._window_size:
            self._wall_times = self._wall_times[-self._window_size :]
        self._tick_count += 1
        self._last_tick_monotonic = now

    def record_skip(self) -> None:
        """Record a skipped tick."""
        self._skip_count += 1

    def record_stale_read(self) -> None:
        """Record a stale data read."""
        self._stale_reads += 1

    def record_exception(self) -> None:
        """Record an exception."""
        self._exceptions += 1

    def last_tick_age_s(self) -> float:
        """Seconds since last tick. Returns inf if never ticked."""
        if self._last_tick_monotonic == 0.0:
            return float("inf")
        return time.monotonic() - self._last_tick_monotonic

    @property
    def loop_hz(self) -> float:
        """Estimated actual loop rate from wall-clock intervals."""
        if len(self._wall_times) < 2:
            return 0.0
        dt = self._wall_times[-1] - self._wall_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._wall_times) - 1) / dt

    @property
    def tick_ms_p50(self) -> float:
        """Median tick duration in milliseconds."""
        if not self._tick_times:
            return 0.0
        ms = [t * 1000.0 for t in self._tick_times]
        return statistics.median(ms)

    @property
    def tick_ms_p95(self) -> float:
        """95th percentile tick duration in milliseconds."""
        if not self._tick_times:
            return 0.0
        ms = sorted(t * 1000.0 for t in self._tick_times)
        idx = int(math.ceil(0.95 * len(ms))) - 1
        return ms[max(0, idx)]

    @property
    def tick_ms_p99(self) -> float:
        """99th percentile tick duration in milliseconds."""
        if not self._tick_times:
            return 0.0
        ms = sorted(t * 1000.0 for t in self._tick_times)
        idx = int(math.ceil(0.99 * len(ms))) - 1
        return ms[max(0, idx)]

    @property
    def total_ticks(self) -> int:
        return self._tick_count

    @property
    def skipped_ticks(self) -> int:
        return self._skip_count

    @property
    def exceptions(self) -> int:
        return self._exceptions

    @property
    def stale_reads(self) -> int:
        return self._stale_reads

    def snapshot(self) -> dict:
        """Return all metrics as a dict for health reporting."""
        return {
            "name": self.name,
            "total_ticks": self._tick_count,
            "skipped_ticks": self._skip_count,
            "exceptions": self._exceptions,
            "stale_reads": self._stale_reads,
            "tick_ms_p50": self.tick_ms_p50,
            "tick_ms_p95": self.tick_ms_p95,
            "tick_ms_p99": self.tick_ms_p99,
            "loop_hz": self.loop_hz,
            "last_tick_age_s": round(self.last_tick_age_s(), 3),
        }
