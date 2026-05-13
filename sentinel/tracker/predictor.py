"""Linear motion predictor for track bbox prediction."""

from __future__ import annotations

from sentinel.common.types import BoundingBox

from .motion import predict_bbox


class LinearPredictor:
    def __init__(self, max_prediction_frames: int = 15) -> None:
        self._max_frames = max_prediction_frames

    def predict(self, track_state: object) -> BoundingBox | None:
        velocity = getattr(track_state, "velocity", None)
        bbox = getattr(track_state, "bbox_xyxy", None)
        lost_frames = getattr(track_state, "lost_frames", 0)

        if velocity is None or bbox is None:
            return None
        if velocity == (0.0, 0.0):
            return None
        if lost_frames > self._max_frames:
            return None

        return predict_bbox(bbox, velocity, lost_frames)
