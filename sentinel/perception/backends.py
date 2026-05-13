"""Perception backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from sentinel.common.types import Detection


@runtime_checkable
class PerceptionBackend(Protocol):
    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp_utc: str) -> list[Detection]: ...
    def close(self) -> None: ...
