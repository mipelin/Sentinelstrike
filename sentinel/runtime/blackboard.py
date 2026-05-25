"""LatestSlot -- generic lock-protected latest-value container."""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Lock-protected container holding only the latest value of type T.

    Write-overwrite semantics: each write() replaces the previous value.
    No history, no FIFO. Callers are responsible for copying mutable
    values (e.g. numpy arrays) before calling write() if they need
    snapshot isolation.

    Thread safety: all public methods acquire self._lock.
    Timing: all timestamps use time.monotonic().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None
        self._seq: int = 0
        self._ts_monotonic: float = 0.0

    def write(self, value: T) -> int:
        """Store value, increment sequence, record monotonic timestamp.

        Returns the new sequence number.
        """
        with self._lock:
            self._value = value
            self._seq += 1
            self._ts_monotonic = time.monotonic()
            return self._seq

    def read(self) -> tuple[T | None, int, float]:
        """Return (value, seq, ts_monotonic) atomically.

        The value is returned by reference (not copied). For mutable
        data like numpy arrays, the caller must copy if snapshot
        isolation is needed.
        """
        with self._lock:
            return self._value, self._seq, self._ts_monotonic

    def age_s(self) -> float:
        """Seconds since last write. Returns inf if never written."""
        with self._lock:
            if self._ts_monotonic == 0.0:
                return float("inf")
            return time.monotonic() - self._ts_monotonic

    def is_stale(self, threshold_s: float) -> bool:
        """True if age_s() > threshold_s."""
        return self.age_s() > threshold_s

    def seq_id(self) -> int:
        """Current sequence number (under lock)."""
        with self._lock:
            return self._seq
