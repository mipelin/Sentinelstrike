"""Tests for IoU computation."""

from sentinel.common.types import BoundingBox
from sentinel.tracker.iou import bbox_area, bbox_iou


def test_bbox_area():
    bb = BoundingBox(x1=0, y1=0, x2=10, y2=20)
    assert bbox_area(bb) == 200


def test_iou_identical_boxes():
    bb = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    assert bbox_iou(bb, bb) == 1.0


def test_iou_no_overlap():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=20, y1=20, x2=30, y2=30)
    assert bbox_iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=5, y1=5, x2=15, y2=15)
    iou = bbox_iou(a, b)
    assert 0 < iou < 1
