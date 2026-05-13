"""Inference stage — runs perception on frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sentinel.common.types import Detection


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)


class InferenceStage:
    """Runs perception backend on a frame to produce detections."""

    def __init__(self, frame_processor: object | None = None) -> None:
        self._frame_processor = frame_processor

    def process(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_utc: str,
    ) -> InferenceResult:
        if self._frame_processor is None:
            return InferenceResult()
        dets = self._frame_processor.process(frame, frame_id=frame_id, timestamp_utc=timestamp_utc)
        return InferenceResult(detections=dets)
