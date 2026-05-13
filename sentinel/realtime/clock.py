"""Clock helpers for the real-time loop."""

from __future__ import annotations

import time
from datetime import UTC, datetime


def monotonic_now_s() -> float:
    return time.monotonic()


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def elapsed_s(start: float) -> float:
    return time.monotonic() - start
