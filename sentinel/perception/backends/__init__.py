"""Perception backend implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from sentinel.common.types import Detection
from sentinel.perception.mock_backend import MockPerceptionBackend
from sentinel.perception.yolo_backend import YoloPerceptionBackend

try:
    from sentinel.perception.backends.tensorrt_backend import TensorRTBackend
except ImportError:
    TensorRTBackend = None  # type: ignore[assignment]


@runtime_checkable
class PerceptionBackend(Protocol):
    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp_utc: str) -> list[Detection]: ...
    def close(self) -> None: ...


__all__ = ["PerceptionBackend", "MockPerceptionBackend", "YoloPerceptionBackend", "TensorRTBackend"]
