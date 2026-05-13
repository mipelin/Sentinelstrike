"""Tracking runner — wraps tracker with event publishing and file output."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.types import Detection, Track
from sentinel.config.schema import TrackerConfig

from .metrics import compute_tracker_metrics
from .prioritization import update_track_priority
from .simple_tracker import SimpleIoUTracker


class TrackingRunner:
    def __init__(
        self,
        config: TrackerConfig,
        mission_id: str,
        event_bus: EventBus | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._run_dir = run_dir
        self._tracker = SimpleIoUTracker(
            iou_threshold=config.iou_threshold,
            max_lost_frames=config.max_lost_frames,
            min_confidence=config.min_confidence,
            prediction_enabled=config.prediction_enabled,
            distance_threshold_px=config.distance_threshold_px,
            iou_weight=config.iou_weight,
            distance_weight=config.distance_weight,
            fps=config.fps,
            stationary_speed_threshold_px_s=config.stationary_speed_threshold_px_s,
            min_hits_to_confirm=config.min_hits_to_confirm,
            publish_tentative_tracks=config.publish_tentative_tracks,
            fast_confirm_confidence=config.fast_confirm_confidence,
            reacquire_window_frames=config.reacquire_window_frames,
            duplicate_distance_px=config.duplicate_distance_px,
            duplicate_iou_threshold=config.duplicate_iou_threshold,
            class_aware_matching=config.class_aware_matching,
            confidence_ema_alpha=config.confidence_ema_alpha,
            bbox_smoothing_enabled=config.bbox_smoothing_enabled,
            bbox_ema_alpha=config.bbox_ema_alpha,
        )
        self._tracks_file = None
        self._prev_status: dict[str, str] = {}
        self._prev_confirmed: dict[str, bool] = {}
        self._frame_count = 0
        self._suppress_after_frames = int(
            config.suppress_stationary_after_s * config.fps
        )
        self._stationary_threshold = config.stationary_speed_threshold_px_s

        if run_dir and config.save_tracks_jsonl:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._tracks_file = (run_dir / "tracks.jsonl").open("w", encoding="utf-8")

    def _publish(self, event_type: str, **payload: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="tracker",
                    event_type=event_type,
                    payload=payload,
                )
            )

    def process_frame_detections(self, frame_id: int, timestamp_utc: str, detections: list[Detection]) -> list[Track]:
        self._frame_count += 1
        self._tracker.update(detections, frame_id, timestamp_utc)

        # Update priority + suppression for internal track states
        for ts in self._tracker._tracks:
            if ts.status != "terminated":
                update_track_priority(
                    ts,
                    suppress_after_frames=self._suppress_after_frames,
                    stationary_threshold_px_s=self._stationary_threshold,
                )

        # Publish events over ALL internal tracks (including tentative)
        for ts in self._tracker._tracks:
            trk = ts.to_public_track()
            prev = self._prev_status.get(trk.track_id)
            is_confirmed = trk.metadata.get("confirmed", True)

            if prev is None:
                self._publish(
                    "track_created",
                    track_id=trk.track_id,
                    class_name=trk.class_name,
                    frame_id=frame_id,
                )
            elif prev == "lost" and trk.status == "active":
                self._publish(
                    "track_reacquired",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                )
            elif prev == "active" and trk.status == "lost":
                self._publish(
                    "track_lost",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                )
            elif prev != "terminated" and trk.status == "terminated":
                self._publish(
                    "track_terminated",
                    track_id=trk.track_id,
                    class_name=trk.class_name,
                    frame_id=frame_id,
                )

            # Confirmation transition
            prev_confirmed = self._prev_confirmed.get(trk.track_id, False)
            if is_confirmed and not prev_confirmed:
                self._publish(
                    "track_confirmed",
                    track_id=trk.track_id,
                    class_name=trk.class_name,
                    frame_id=frame_id,
                )

            self._prev_status[trk.track_id] = trk.status
            self._prev_confirmed[trk.track_id] = is_confirmed

            # Predicted track event
            if trk.metadata.get("predicted_bbox") is not None:
                self._publish(
                    "track_predicted",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                    prediction_age=trk.metadata.get("prediction_age", 0),
                )

            if self._tracks_file:
                self._tracks_file.write(trk.model_dump_json() + "\n")

        frame_tracks = self._tracker.get_frame_active_tracks()
        all_tracks = self._tracker.get_all_tracks()
        self._publish(
            "tracker_updated",
            frame_id=frame_id,
            track_count=len(all_tracks),
            frame_active_count=len(frame_tracks),
            active_tracks=sum(1 for trk in all_tracks if trk.status == "active"),
            lost_tracks=sum(1 for trk in all_tracks if trk.status == "lost"),
        )
        return frame_tracks

    def close(self, frame_count: int | None = None) -> dict:
        if self._tracks_file:
            self._tracks_file.close()

        fc = frame_count if frame_count is not None else self._frame_count
        all_tracks = self._tracker.get_all_tracks()
        metrics = compute_tracker_metrics(
            all_tracks, fc,
            duplicate_suppressed_count=self._tracker.duplicate_suppressed_count,
            reacquisition_from_lost_count=self._tracker.reacquisition_from_lost_count,
        )

        if self._run_dir:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            (self._run_dir / "tracker_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        self._publish("tracking_completed", **metrics)
        logger.info(
            "Tracking completed: {} total tracks, {} confirmed, {} tentative",
            metrics["total_tracks"],
            metrics.get("confirmed_tracks", 0),
            metrics.get("tentative_tracks", 0),
        )
        return metrics

    def get_all_tracks(self) -> list[Track]:
        return self._tracker.get_all_tracks()

    def get_tentative_tracks(self) -> list[Track]:
        return self._tracker.get_tentative_tracks()
