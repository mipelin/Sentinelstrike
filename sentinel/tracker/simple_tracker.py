"""IoU + distance-based multi-object tracker with motion prediction."""

from __future__ import annotations

import math

from loguru import logger

from sentinel.common.types import BoundingBox, Detection, Track

from .association import build_score_matrix
from .iou import bbox_center, bbox_iou
from .motion import center_distance, predict_bbox
from .track import ConfirmationState, TrackState


class SimpleIoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        min_confidence: float = 0.0,
        prediction_enabled: bool = False,
        distance_threshold_px: float = 100.0,
        iou_weight: float = 1.0,
        distance_weight: float = 0.0,
        fps: float = 5.0,
        stationary_speed_threshold_px_s: float = 3.0,
        min_hits_to_confirm: int = 1,
        publish_tentative_tracks: bool = True,
        fast_confirm_confidence: float = 0.85,
        reacquire_window_frames: int = 20,
        duplicate_distance_px: float = 80.0,
        duplicate_iou_threshold: float = 0.1,
        class_aware_matching: bool = True,
        confidence_ema_alpha: float = 0.35,
        bbox_smoothing_enabled: bool = False,
        bbox_ema_alpha: float = 0.35,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._min_confidence = min_confidence
        self._prediction_enabled = prediction_enabled
        self._distance_threshold_px = distance_threshold_px
        self._iou_weight = iou_weight
        self._distance_weight = distance_weight
        self._fps = fps
        self._stationary_speed_threshold_px_s = stationary_speed_threshold_px_s
        self._min_hits_to_confirm = min_hits_to_confirm
        self._publish_tentative_tracks = publish_tentative_tracks
        self._fast_confirm_confidence = fast_confirm_confidence
        self._reacquire_window_frames = reacquire_window_frames
        self._duplicate_distance_px = duplicate_distance_px
        self._duplicate_iou_threshold = duplicate_iou_threshold
        self._class_aware_matching = class_aware_matching
        self._confidence_ema_alpha = confidence_ema_alpha
        self._bbox_smoothing_enabled = bbox_smoothing_enabled
        self._bbox_ema_alpha = bbox_ema_alpha
        self._tracks: list[TrackState] = []
        self._next_id = 1
        self._frame_active_ids: set[str] = set()
        self._duplicate_suppressed_count: int = 0
        self._reacquisition_from_lost_count: int = 0

    def _make_id(self) -> str:
        tid = f"trk_{self._next_id:06d}"
        self._next_id += 1
        return tid

    def _get_association_bbox(self, trk: TrackState) -> BoundingBox:
        if self._prediction_enabled and trk.predicted_bbox is not None:
            return trk.predicted_bbox
        return trk.bbox_xyxy

    def _find_reacquisition_target(
        self,
        det_bbox: BoundingBox,
        det_class: str,
        frame_id: int,
    ) -> TrackState | None:
        """Check recently-lost/terminated tracks for re-association."""
        det_cx, det_cy = bbox_center(det_bbox)
        best_trk: TrackState | None = None
        best_dist: float = float("inf")

        for trk in self._tracks:
            if trk.status not in ("lost", "terminated"):
                continue
            frames_since = frame_id - trk.last_frame_id
            if frames_since > self._reacquire_window_frames:
                continue
            if trk.class_name != det_class:
                continue

            # Check distance from last known or predicted bbox
            check_bbox = trk.predicted_bbox or trk.bbox_xyxy
            trk_cx, trk_cy = bbox_center(check_bbox)
            dist = math.sqrt((trk_cx - det_cx) ** 2 + (trk_cy - det_cy) ** 2)
            if dist > self._duplicate_distance_px:
                continue

            # Also accept if IoU overlap exists
            iou = bbox_iou(det_bbox, trk.bbox_xyxy)
            if iou < self._duplicate_iou_threshold and dist >= self._duplicate_distance_px:
                continue

            if dist < best_dist:
                best_dist = dist
                best_trk = trk

        return best_trk

    def _check_confirm(self, trk: TrackState, det_confidence: float) -> None:
        """Check if a track should be promoted from TENTATIVE to CONFIRMED."""
        if trk.confirmation_state == ConfirmationState.CONFIRMED:
            return
        if trk.total_hits >= self._min_hits_to_confirm:
            trk.confirmation_state = ConfirmationState.CONFIRMED
        elif det_confidence >= self._fast_confirm_confidence and trk.total_hits >= 1:
            trk.confirmation_state = ConfirmationState.CONFIRMED

    def update(self, detections: list[Detection], frame_id: int, timestamp_utc: str) -> list[Track]:
        self._frame_active_ids = set()
        filtered = [d for d in detections if d.confidence >= self._min_confidence]

        candidates = [t for t in self._tracks if t.status in ("active", "lost")]
        matched_track_idx: set[int] = set()
        matched_det_idx: set[int] = set()

        # --- STEP 1: Match with active/lost candidates ---
        if candidates and filtered:
            candidate_bboxes = [self._get_association_bbox(t) for t in candidates]
            candidate_classes = [t.class_name for t in candidates]
            det_classes = [d.class_name for d in filtered]

            score_matrix = build_score_matrix(
                detections=filtered,
                candidate_bboxes=candidate_bboxes,
                iou_weight=self._iou_weight,
                distance_weight=self._distance_weight,
                distance_threshold=self._distance_threshold_px,
                class_names=candidate_classes if self._class_aware_matching else None,
                detection_classes=det_classes if self._class_aware_matching else None,
            )

            # Greedy matching — pick highest score above threshold
            for _round in range(min(len(filtered), len(candidates))):
                best_score = self._iou_threshold
                best_ti = -1
                best_di = -1
                for di in range(len(filtered)):
                    if di in matched_det_idx:
                        continue
                    for ti in range(len(candidates)):
                        if ti in matched_track_idx:
                            continue
                        s = score_matrix[di][ti]
                        if s > best_score:
                            best_score = s
                            best_ti = ti
                            best_di = di
                if best_ti < 0:
                    break
                matched_track_idx.add(best_ti)
                matched_det_idx.add(best_di)

                trk = candidates[best_ti]
                det = filtered[best_di]
                was_lost = trk.status == "lost"

                # Velocity estimation
                old_center = bbox_center(trk.bbox_xyxy)
                new_center = bbox_center(det.bbox_xyxy)
                trk.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                trk.prev_bbox_xyxy = trk.bbox_xyxy

                # Prediction error if we had a prediction
                if trk.predicted_bbox is not None:
                    trk.last_prediction_error_px = center_distance(trk.predicted_bbox, det.bbox_xyxy)

                # Reacquisition tracking
                if trk.consecutive_misses > 0:
                    trk.reacquired_count += 1

                trk.bbox_xyxy = det.bbox_xyxy
                trk.confidence = det.confidence
                trk.age_frames += 1
                trk.total_hits += 1
                trk.lost_frames = 0
                trk.status = "active"
                trk.last_seen_utc = timestamp_utc
                trk.last_frame_id = frame_id
                trk.source_detection_id = det.detection_id
                trk.consecutive_hits += 1
                trk.consecutive_misses = 0
                trk.predicted_bbox = None
                trk.prediction_age = 0
                trk.push_history(new_center[0], new_center[1])
                trk.last_transition = "reacquired" if was_lost else "updated"
                trk.update_motion(
                    new_center[0], new_center[1],
                    fps=self._fps,
                    stationary_threshold_px_s=self._stationary_speed_threshold_px_s,
                )
                trk.update_smoothing(
                    det.bbox_xyxy, det.confidence,
                    bbox_alpha=self._bbox_ema_alpha,
                    conf_alpha=self._confidence_ema_alpha,
                    bbox_smoothing_enabled=self._bbox_smoothing_enabled,
                )
                self._check_confirm(trk, det.confidence)
                trk.compute_quality_score()
                self._frame_active_ids.add(trk.track_id)
        elif candidates:
            pass

        # --- STEP 2: Duplicate suppression — check lost/terminated for unmatched detections ---
        if filtered:
            unmatched_dets = [
                di for di in range(len(filtered))
                if di not in matched_det_idx
            ]
            truly_unmatched: list[int] = []
            for di in unmatched_dets:
                det = filtered[di]
                reacq = self._find_reacquisition_target(det.bbox_xyxy, det.class_name, frame_id)
                if reacq is not None:
                    # Reacquire this track instead of creating new
                    old_center = bbox_center(reacq.bbox_xyxy)
                    new_center = bbox_center(det.bbox_xyxy)
                    reacq.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                    reacq.prev_bbox_xyxy = reacq.bbox_xyxy
                    reacq.bbox_xyxy = det.bbox_xyxy
                    reacq.confidence = det.confidence
                    reacq.age_frames += 1
                    reacq.total_hits += 1
                    reacq.lost_frames = 0
                    reacq.status = "active"
                    reacq.last_seen_utc = timestamp_utc
                    reacq.last_frame_id = frame_id
                    reacq.source_detection_id = det.detection_id
                    reacq.consecutive_hits = 1
                    reacq.consecutive_misses = 0
                    reacq.reacquired_count += 1
                    reacq.predicted_bbox = None
                    reacq.prediction_age = 0
                    reacq.push_history(new_center[0], new_center[1])
                    reacq.last_transition = "reacquired"
                    reacq.update_motion(
                        new_center[0], new_center[1],
                        fps=self._fps,
                        stationary_threshold_px_s=self._stationary_speed_threshold_px_s,
                    )
                    reacq.update_smoothing(
                        det.bbox_xyxy, det.confidence,
                        bbox_alpha=self._bbox_ema_alpha,
                        conf_alpha=self._confidence_ema_alpha,
                        bbox_smoothing_enabled=self._bbox_smoothing_enabled,
                    )
                    self._check_confirm(reacq, det.confidence)
                    reacq.compute_quality_score()
                    self._frame_active_ids.add(reacq.track_id)
                    self._duplicate_suppressed_count += 1
                    self._reacquisition_from_lost_count += 1
                else:
                    truly_unmatched.append(di)

            # --- STEP 3: Create new tracks for truly unmatched detections ---
            for di in truly_unmatched:
                det = filtered[di]
                ts = TrackState(
                    track_id=self._make_id(),
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                    first_seen_utc=timestamp_utc,
                    source_detection_id=det.detection_id,
                )
                cx, cy = bbox_center(det.bbox_xyxy)
                ts.last_frame_id = frame_id
                ts.push_history(cx, cy)
                ts.update_motion(
                    cx, cy,
                    fps=self._fps,
                    stationary_threshold_px_s=self._stationary_speed_threshold_px_s,
                )
                # Fast-confirm for high confidence
                if det.confidence >= self._fast_confirm_confidence:
                    ts.confirmation_state = ConfirmationState.CONFIRMED
                ts.compute_quality_score()
                self._tracks.append(ts)
                self._frame_active_ids.add(ts.track_id)

        # --- STEP 4: Age unmatched candidate tracks ---
        for ti, trk in enumerate(candidates):
            if ti not in matched_track_idx:
                trk.lost_frames += 1
                trk.age_frames += 1
                trk.consecutive_hits = 0
                trk.consecutive_misses += 1
                trk.total_misses += 1

                # Prediction for lost tracks
                if self._prediction_enabled and trk.velocity != (0.0, 0.0):
                    trk.predicted_bbox = predict_bbox(
                        trk.bbox_xyxy, trk.velocity, trk.lost_frames,
                    )
                    trk.prediction_age += 1
                    pcx, pcy = bbox_center(trk.predicted_bbox)
                    trk.push_history(pcx, pcy)
                else:
                    trk.predicted_bbox = None

                if trk.lost_frames > self._max_lost_frames:
                    trk.status = "terminated"
                    trk.last_transition = "terminated"
                else:
                    trk.status = "lost"
                    trk.last_transition = "lost"
                trk.last_frame_id = frame_id
                trk.compute_quality_score()

        # --- STEP 5: Return filtered tracks ---
        if self._publish_tentative_tracks:
            return [t.to_public_track() for t in self._tracks if t.status != "terminated"]
        else:
            return [
                t.to_public_track() for t in self._tracks
                if t.status != "terminated"
                and t.confirmation_state == ConfirmationState.CONFIRMED
            ]

    def get_active_tracks(self) -> list[Track]:
        return [t.to_public_track() for t in self._tracks if t.status == "active"]

    def get_frame_active_tracks(self) -> list[Track]:
        return [t.to_public_track() for t in self._tracks if t.track_id in self._frame_active_ids]

    def get_all_tracks(self) -> list[Track]:
        return [t.to_public_track() for t in self._tracks]

    def get_confirmed_tracks(self) -> list[Track]:
        return [
            t.to_public_track() for t in self._tracks
            if t.status != "terminated"
            and t.confirmation_state == ConfirmationState.CONFIRMED
        ]

    def get_tentative_tracks(self) -> list[Track]:
        return [
            t.to_public_track() for t in self._tracks
            if t.status != "terminated"
            and t.confirmation_state == ConfirmationState.TENTATIVE
        ]

    @property
    def duplicate_suppressed_count(self) -> int:
        return self._duplicate_suppressed_count

    @property
    def reacquisition_from_lost_count(self) -> int:
        return self._reacquisition_from_lost_count

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
