"""Mock perception backend — deterministic dummy detections for testing."""

from __future__ import annotations

import numpy as np

from sentinel.common.types import BoundingBox, Detection


class MockPerceptionBackend:
    def __init__(self, classes: list[str] | None = None, confidence_threshold: float = 0.35) -> None:
        self._classes = classes or ["object"]
        self._confidence_threshold = confidence_threshold

    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp_utc: str) -> list[Detection]:
        if frame_id % 10 != 0:
            return []

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_w, box_h = max(w // 8, 20), max(h // 8, 20)
        conf = max(self._confidence_threshold, 0.75)

        return [
            Detection(
                frame_id=frame_id,
                timestamp_utc=timestamp_utc,
                class_name=self._classes[0],
                confidence=conf,
                bbox_xyxy=BoundingBox(
                    x1=max(cx - box_w, 0),
                    y1=max(cy - box_h, 0),
                    x2=min(cx + box_w, w - 1),
                    y2=min(cy + box_h, h - 1),
                ),
                source="mock",
            )
        ]

    def close(self) -> None:
        pass
