"""Rate limiter for the real-time loop."""

from __future__ import annotations

import time

from loguru import logger


class RateLimiter:
    """Enforces a target tick rate with drift-free scheduling.

    Uses an anchor-based approach: each tick targets anchor + N * period.
    This prevents cumulative drift from processing jitter.
    """

    def __init__(self, target_hz: float = 5.0, sleep_enabled: bool = True) -> None:
        if target_hz <= 0:
            raise ValueError(f"target_hz must be > 0, got {target_hz}")
        self._target_hz = target_hz
        self._sleep_enabled = sleep_enabled
        self._period_s = 1.0 / target_hz
        self._anchor: float | None = None
        self._tick_count: int = 0

    @property
    def target_hz(self) -> float:
        return self._target_hz

    @property
    def period_s(self) -> float:
        return self._period_s

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def start(self) -> None:
        """Initialize the timing anchor. Call once before the loop."""
        self._anchor = time.monotonic()
        self._tick_count = 0

    def wait(self) -> None:
        """Wait until the next scheduled tick.

        Sleeps for the remaining time after processing. Uses anchor-based
        scheduling to prevent cumulative drift.
        """
        self._tick_count += 1

        if self._anchor is None:
            self._anchor = time.monotonic()
            return

        if not self._sleep_enabled:
            self._anchor = time.monotonic()
            return

        next_tick_time = self._anchor + self._period_s * self._tick_count
        now = time.monotonic()
        remaining = next_tick_time - now

        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -self._period_s:
            # Fell behind by more than one full period — reset anchor to avoid burst catch-up
            logger.warning("Rate limiter fell behind by {:.1f}ms, resetting anchor", -remaining * 1000)
            self._anchor = time.monotonic()
            self._tick_count = 0

    def reset(self) -> None:
        """Reset timing state for reuse."""
        self._anchor = None
        self._tick_count = 0
