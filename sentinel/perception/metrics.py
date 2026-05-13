"""Perception metrics computation."""

from __future__ import annotations

from collections import Counter

from sentinel.common.types import Detection


def compute_perception_metrics(
    frame_count: int,
    detections: list[Detection],
    started_at_utc: str,
    ended_at_utc: str,
) -> dict:
    class_counts = Counter(d.class_name for d in detections)
    avg_conf = sum(d.confidence for d in detections) / len(detections) if detections else 0.0
    return {
        "frame_count": frame_count,
        "detection_count": len(detections),
        "classes": dict(class_counts),
        "average_confidence": round(avg_conf, 4),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
    }
