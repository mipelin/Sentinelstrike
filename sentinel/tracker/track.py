"""Internal track state for the tracker."""

from __future__ import annotations

import math
from enum import Enum

from sentinel.common.types import BoundingBox, Track


class MovementState(str, Enum):
    STATIONARY = "STATIONARY"
    SLOW_MOVING = "SLOW_MOVING"
    MOVING = "MOVING"
    FAST_MOVING = "FAST_MOVING"


class ConfirmationState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"


class TrackState:
    def __init__(
        self,
        track_id: str,
        class_name: str,
        confidence: float,
        bbox_xyxy: BoundingBox,
        first_seen_utc: str,
        source_detection_id: str | None = None,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_xyxy = bbox_xyxy
        self.first_seen_utc = first_seen_utc
        self.last_seen_utc = first_seen_utc
        self.age_frames = 1
        self.lost_frames = 0
        self.status = "active"
        self.source_detection_id = source_detection_id

        # Motion / prediction fields
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.prev_bbox_xyxy: BoundingBox | None = None
        self.predicted_bbox: BoundingBox | None = None
        self.prediction_age: int = 0
        self.consecutive_hits: int = 1
        self.consecutive_misses: int = 0
        self.reacquired_count: int = 0
        self.last_prediction_error_px: float | None = None
        self.last_frame_id: int = 0
        self.history_px: list[tuple[float, float]] = []
        self.last_transition: str = "created"

        # Motion classification fields
        self.instantaneous_speed_px_s: float = 0.0
        self.ema_speed_px_s: float = 0.0
        self.displacement_px: float = 0.0
        self.stationary_frames: int = 0
        self.moving_frames: int = 0
        self.movement_state: MovementState = MovementState.STATIONARY
        self._initial_center: tuple[float, float] | None = None

        # Priority / suppression
        self.priority_score: float = 0.0
        self.priority_level: str = "LOW"
        self.suppressed: bool = False

        # Confirmation state
        self.confirmation_state: ConfirmationState = ConfirmationState.TENTATIVE
        self.total_hits: int = 1
        self.total_misses: int = 0
        self.hit_rate: float = 1.0
        self.track_quality_score: float = 0.0

        # Smoothing
        self.smoothed_confidence: float = confidence
        self.display_bbox_xyxy: BoundingBox | None = None

    def update_motion(
        self,
        new_cx: float,
        new_cy: float,
        fps: float = 5.0,
        stationary_threshold_px_s: float = 3.0,
        ema_alpha: float = 0.3,
    ) -> None:
        """Update motion classification based on new center position."""
        if self._initial_center is None:
            self._initial_center = (new_cx, new_cy)

        # Displacement from initial position
        dx = new_cx - self._initial_center[0]
        dy = new_cy - self._initial_center[1]
        self.displacement_px = math.sqrt(dx * dx + dy * dy)

        # Instantaneous speed from velocity
        vx, vy = self.velocity
        frame_speed = math.sqrt(vx * vx + vy * vy)
        dt = 1.0 / max(fps, 0.1)
        self.instantaneous_speed_px_s = frame_speed / dt

        # EMA smoothing
        if self.ema_speed_px_s == 0.0:
            self.ema_speed_px_s = self.instantaneous_speed_px_s
        else:
            self.ema_speed_px_s = ema_alpha * self.instantaneous_speed_px_s + (1 - ema_alpha) * self.ema_speed_px_s

        # Classify movement
        speed = self.ema_speed_px_s
        if speed < stationary_threshold_px_s:
            self.movement_state = MovementState.STATIONARY
            self.stationary_frames += 1
        elif speed < stationary_threshold_px_s * 3:
            self.movement_state = MovementState.SLOW_MOVING
            self.moving_frames += 1
        elif speed < stationary_threshold_px_s * 10:
            self.movement_state = MovementState.MOVING
            self.moving_frames += 1
        else:
            self.movement_state = MovementState.FAST_MOVING
            self.moving_frames += 1

    def update_smoothing(
        self,
        det_bbox: BoundingBox,
        det_confidence: float,
        bbox_alpha: float = 0.35,
        conf_alpha: float = 0.35,
        bbox_smoothing_enabled: bool = False,
    ) -> None:
        """EMA-smooth bbox and confidence. Raw bbox preserved for association."""
        # Confidence smoothing
        self.smoothed_confidence = conf_alpha * det_confidence + (1 - conf_alpha) * self.smoothed_confidence

        # Bbox smoothing
        if bbox_smoothing_enabled:
            prev = self.display_bbox_xyxy or det_bbox
            self.display_bbox_xyxy = BoundingBox(
                x1=max(int(round(bbox_alpha * det_bbox.x1 + (1 - bbox_alpha) * prev.x1)), 0),
                y1=max(int(round(bbox_alpha * det_bbox.y1 + (1 - bbox_alpha) * prev.y1)), 0),
                x2=max(int(round(bbox_alpha * det_bbox.x2 + (1 - bbox_alpha) * prev.x2)), prev.x1 + 1),
                y2=max(int(round(bbox_alpha * det_bbox.y2 + (1 - bbox_alpha) * prev.y2)), prev.y1 + 1),
            )
        else:
            self.display_bbox_xyxy = None

    def compute_quality_score(self) -> None:
        """Compute track quality 0-100 based on hit_rate, age, confidence, confirmation."""
        if self.age_frames == 0:
            self.track_quality_score = 0.0
            return
        self.hit_rate = self.total_hits / self.age_frames
        hit_component = min(self.hit_rate * 40.0, 40.0)
        age_component = min(self.age_frames / 50.0 * 20.0, 20.0)
        conf_component = self.smoothed_confidence * 20.0
        confirm_component = 20.0 if self.confirmation_state == ConfirmationState.CONFIRMED else 0.0
        self.track_quality_score = min(hit_component + age_component + conf_component + confirm_component, 100.0)

    def push_history(self, cx: float, cy: float, max_points: int = 20) -> None:
        self.history_px.append((round(cx, 2), round(cy, 2)))
        if len(self.history_px) > max_points:
            self.history_px = self.history_px[-max_points:]

    def to_public_track(self) -> Track:
        meta: dict = {}
        meta["velocity"] = list(self.velocity)
        meta["predicted_bbox"] = (
            {"x1": self.predicted_bbox.x1, "y1": self.predicted_bbox.y1,
             "x2": self.predicted_bbox.x2, "y2": self.predicted_bbox.y2}
            if self.predicted_bbox else None
        )
        meta["prediction_age"] = self.prediction_age
        meta["consecutive_hits"] = self.consecutive_hits
        meta["consecutive_misses"] = self.consecutive_misses
        meta["reacquired_count"] = self.reacquired_count
        meta["frame_id"] = self.last_frame_id
        meta["history_px"] = [[x, y] for x, y in self.history_px]
        meta["last_transition"] = self.last_transition
        if self.last_prediction_error_px is not None:
            meta["last_prediction_error_px"] = round(self.last_prediction_error_px, 2)

        # Motion classification
        meta["average_speed_px_s"] = round(self.ema_speed_px_s, 2)
        meta["displacement_px"] = round(self.displacement_px, 2)
        meta["stationary_frames"] = self.stationary_frames
        meta["moving_frames"] = self.moving_frames
        meta["movement_state"] = self.movement_state.value

        # Priority
        meta["priority_score"] = round(self.priority_score, 2)
        meta["priority_level"] = self.priority_level
        meta["suppressed"] = self.suppressed

        # Confirmation + quality
        meta["confirmation_state"] = self.confirmation_state.value
        meta["confirmed"] = self.confirmation_state == ConfirmationState.CONFIRMED
        meta["total_hits"] = self.total_hits
        meta["total_misses"] = self.total_misses
        meta["hit_rate"] = round(self.hit_rate, 3)
        meta["smoothed_confidence"] = round(self.smoothed_confidence, 4)
        meta["track_quality_score"] = round(self.track_quality_score, 1)

        # Use display bbox for confirmed tracks with smoothing
        publish_bbox = self.bbox_xyxy
        if self.display_bbox_xyxy is not None:
            publish_bbox = self.display_bbox_xyxy

        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox_xyxy=publish_bbox,
            status=self.status,
            age_frames=self.age_frames,
            last_seen_utc=self.last_seen_utc,
            first_seen_utc=self.first_seen_utc,
            lost_frames=self.lost_frames,
            source_detection_id=self.source_detection_id,
            metadata=meta,
        )
