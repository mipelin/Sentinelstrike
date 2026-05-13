"""Tests for mock perception backend."""

import numpy as np

from sentinel.common.time import utc_now_iso
from sentinel.perception.mock_backend import MockPerceptionBackend


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detects_on_multiples_of_10():
    backend = MockPerceptionBackend(classes=["person"])
    ts = utc_now_iso()
    dets = backend.detect_frame(_frame(), frame_id=0, timestamp_utc=ts)
    assert len(dets) == 1
    assert dets[0].class_name == "person"


def test_no_detection_on_frame_1():
    backend = MockPerceptionBackend(classes=["person"])
    dets = backend.detect_frame(_frame(), frame_id=1, timestamp_utc=utc_now_iso())
    assert len(dets) == 0


def test_bbox_within_frame():
    backend = MockPerceptionBackend(classes=["person"])
    frame = _frame(h=240, w=320)
    det = backend.detect_frame(frame, frame_id=0, timestamp_utc=utc_now_iso())[0]
    assert det.bbox_xyxy.x1 >= 0
    assert det.bbox_xyxy.y1 >= 0
    assert det.bbox_xyxy.x2 < 320
    assert det.bbox_xyxy.y2 < 240


def test_confidence_above_threshold():
    backend = MockPerceptionBackend(classes=["car"], confidence_threshold=0.5)
    det = backend.detect_frame(_frame(), frame_id=0, timestamp_utc=utc_now_iso())[0]
    assert det.confidence >= 0.5
