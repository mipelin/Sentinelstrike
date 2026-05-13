"""Tests for perception metrics."""

from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, Detection
from sentinel.perception.metrics import compute_perception_metrics


def _det(class_name: str, confidence: float) -> Detection:
    return Detection(
        frame_id=0,
        timestamp_utc=utc_now_iso(),
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=BoundingBox(x1=0, y1=0, x2=10, y2=10),
    )


def test_metrics_with_detections():
    dets = [_det("car", 0.8), _det("person", 0.6), _det("car", 0.9)]
    m = compute_perception_metrics(10, dets, "t0", "t1")
    assert m["frame_count"] == 10
    assert m["detection_count"] == 3
    assert m["classes"]["car"] == 2
    assert m["classes"]["person"] == 1
    assert abs(m["average_confidence"] - (0.8 + 0.6 + 0.9) / 3) < 0.01


def test_metrics_without_detections():
    m = compute_perception_metrics(5, [], "t0", "t1")
    assert m["detection_count"] == 0
    assert m["average_confidence"] == 0
    assert m["classes"] == {}


def test_class_counts():
    dets = [_det("bus", 0.7)] * 5
    m = compute_perception_metrics(1, dets, "t0", "t1")
    assert m["classes"]["bus"] == 5
