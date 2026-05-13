"""Tracking stage — detections to tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.common.types import Detection, Track


@dataclass
class TrackingResult:
    tracks: list[Track] = field(default_factory=list)


class TrackingStage:
    """Processes detections through the tracker."""

    def __init__(self, tracker_runner: object | None = None) -> None:
        self._tracker_runner = tracker_runner

    def process(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp_utc: str,
    ) -> TrackingResult:
        if self._tracker_runner is None:
            return TrackingResult()
        tracks = self._tracker_runner.process_frame_detections(
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
            detections=detections,
        )
        return TrackingResult(tracks=tracks)
