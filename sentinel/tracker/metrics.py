"""Tracker metrics computation."""

from __future__ import annotations

import math
from collections import Counter

from sentinel.common.types import Track


def compute_tracker_metrics(
    tracks: list[Track],
    frame_count: int,
    *,
    duplicate_suppressed_count: int = 0,
    reacquisition_from_lost_count: int = 0,
) -> dict:
    total = len(tracks)
    active = sum(1 for t in tracks if t.status == "active")
    lost = sum(1 for t in tracks if t.status == "lost")
    terminated = sum(1 for t in tracks if t.status == "terminated")
    class_counts = Counter(t.class_name for t in tracks)
    avg_age = sum(t.age_frames for t in tracks) / total if total else 0

    reacquired_count = sum(t.metadata.get("reacquired_count", 0) for t in tracks)

    prediction_errors = [
        t.metadata["last_prediction_error_px"]
        for t in tracks
        if t.metadata.get("last_prediction_error_px") is not None
    ]
    avg_pred_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0

    # ID switches estimate: terminated tracks where a same-class track was created nearby
    id_switches = 0
    terminated_tracks = [t for t in tracks if t.status == "terminated"]
    other_tracks = [t for t in tracks if t.status != "terminated"]
    for tt in terminated_tracks:
        if tt.bbox_xyxy is None:
            continue
        ttc = bbox_center_of_track(tt)
        for ot in other_tracks:
            if ot.class_name != tt.class_name or ot.bbox_xyxy is None:
                continue
            otc = bbox_center_of_track(ot)
            dist = math.sqrt((ttc[0] - otc[0]) ** 2 + (ttc[1] - otc[1]) ** 2)
            if dist < 100:
                id_switches += 1
                break

    # Motion + priority metrics
    moving_tracks = sum(
        1 for t in tracks
        if t.metadata.get("movement_state") in ("MOVING", "FAST_MOVING", "SLOW_MOVING")
    )
    stationary_tracks = sum(
        1 for t in tracks
        if t.metadata.get("movement_state") == "STATIONARY"
    )
    suppressed_stationary_tracks = sum(
        1 for t in tracks
        if t.metadata.get("suppressed") is True
    )
    speeds = [
        t.metadata["average_speed_px_s"]
        for t in tracks
        if t.metadata.get("average_speed_px_s") is not None
    ]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    high_priority_tracks = sum(
        1 for t in tracks
        if t.metadata.get("priority_level") in ("HIGH", "CRITICAL")
    )

    # Confirmation + quality metrics
    confirmed_tracks = sum(
        1 for t in tracks
        if t.metadata.get("confirmed") is True
    )
    tentative_tracks = sum(
        1 for t in tracks
        if t.status != "terminated" and t.metadata.get("confirmed") is False
    )
    quality_scores = [
        t.metadata["track_quality_score"]
        for t in tracks
        if t.metadata.get("track_quality_score") is not None
    ]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    id_churn_rate = round(total / max(frame_count, 1), 4)

    return {
        "frame_count": frame_count,
        "total_tracks": total,
        "active_tracks": active,
        "lost_tracks": lost,
        "terminated_tracks": terminated,
        "classes": dict(class_counts),
        "average_track_age_frames": round(avg_age, 2),
        "reacquired_tracks": reacquired_count,
        "average_track_lifetime_frames": round(avg_age, 2),
        "average_prediction_error_px": round(avg_pred_error, 2),
        "id_switches_estimate": id_switches,
        "moving_tracks": moving_tracks,
        "stationary_tracks": stationary_tracks,
        "suppressed_stationary_tracks": suppressed_stationary_tracks,
        "average_track_speed_px_s": round(avg_speed, 2),
        "high_priority_tracks": high_priority_tracks,
        "confirmed_tracks": confirmed_tracks,
        "tentative_tracks": tentative_tracks,
        "unique_tracks": total,
        "id_churn_rate": id_churn_rate,
        "average_track_quality": round(avg_quality, 2),
        "duplicate_suppressed_count": duplicate_suppressed_count,
        "reacquisition_from_lost_count": reacquisition_from_lost_count,
    }


def bbox_center_of_track(t: Track) -> tuple[float, float]:
    if t.bbox_xyxy is None:
        return (0.0, 0.0)
    return ((t.bbox_xyxy.x1 + t.bbox_xyxy.x2) / 2.0, (t.bbox_xyxy.y1 + t.bbox_xyxy.y2) / 2.0)
