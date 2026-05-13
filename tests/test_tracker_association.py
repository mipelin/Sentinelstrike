"""Tests for hybrid association scoring."""

from sentinel.common.types import BoundingBox, Detection
from sentinel.tracker.association import build_score_matrix, compute_hybrid_score


def _det(x1=0, y1=0, x2=10, y2=10, cls="person", conf=0.9, fid=0):
    return Detection(
        frame_id=fid,
        timestamp_utc="2026-01-01T00:00:00Z",
        class_name=cls,
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_hybrid_score_pure_iou():
    # When distance_weight=0, score = iou_weight * iou
    score = compute_hybrid_score(iou=0.5, center_dist=100.0, distance_threshold=100.0, iou_weight=1.0, distance_weight=0.0)
    assert abs(score - 0.5) < 0.001


def test_hybrid_score_pure_distance():
    # When iou_weight=0, score = distance_weight * normalized_dist
    score = compute_hybrid_score(iou=0.5, center_dist=50.0, distance_threshold=100.0, iou_weight=0.0, distance_weight=1.0)
    assert abs(score - 0.5) < 0.001  # 1.0 - 50/100 = 0.5


def test_hybrid_score_zero_distance():
    score = compute_hybrid_score(iou=0.3, center_dist=0.0, distance_threshold=100.0, iou_weight=0.5, distance_weight=0.5)
    # 0.5 * 0.3 + 0.5 * 1.0 = 0.65
    assert abs(score - 0.65) < 0.001


def test_hybrid_score_far_distance():
    score = compute_hybrid_score(iou=0.0, center_dist=200.0, distance_threshold=100.0, iou_weight=0.5, distance_weight=0.5)
    # 0.5 * 0.0 + 0.5 * 0.0 = 0.0 (dist exceeds threshold → normalized = 0)
    assert abs(score - 0.0) < 0.001


def test_hybrid_score_zero_threshold():
    score = compute_hybrid_score(iou=0.5, center_dist=10.0, distance_threshold=0.0, iou_weight=0.5, distance_weight=0.5)
    # distance component = 0, score = 0.5 * 0.5 = 0.25
    assert abs(score - 0.25) < 0.001


def test_build_score_matrix_basic():
    dets = [_det(x1=0, y1=0, x2=10, y2=10)]
    bboxes = [BoundingBox(x1=1, y1=1, x2=11, y2=11)]
    matrix = build_score_matrix(dets, bboxes, iou_weight=1.0, distance_weight=0.0, distance_threshold=100.0)
    assert len(matrix) == 1
    assert len(matrix[0]) == 1
    assert matrix[0][0] > 0  # High IoU overlap


def test_build_score_matrix_class_filter():
    dets = [_det(cls="person"), _det(cls="car")]
    bboxes = [BoundingBox(x1=0, y1=0, x2=10, y2=10)]
    matrix = build_score_matrix(
        dets, bboxes, 1.0, 0.0, 100.0,
        class_names=["person"],
        detection_classes=["person", "car"],
    )
    assert matrix[0][0] > 0   # person matches person
    assert matrix[1][0] == 0.0  # car doesn't match person


def test_build_score_matrix_no_overlap():
    dets = [_det(x1=0, y1=0, x2=10, y2=10)]
    bboxes = [BoundingBox(x1=500, y1=500, x2=510, y2=510)]
    matrix = build_score_matrix(dets, bboxes, 1.0, 0.0, 100.0)
    assert matrix[0][0] == 0.0


def test_build_score_matrix_hybrid():
    dets = [_det(x1=0, y1=0, x2=10, y2=10)]
    # Close center but no overlap
    bboxes = [BoundingBox(x1=12, y1=12, x2=22, y2=22)]
    # Pure IoU → 0
    m_iou = build_score_matrix(dets, bboxes, 1.0, 0.0, 100.0)
    assert m_iou[0][0] == 0.0
    # With distance weight → should be > 0 (centers are close)
    m_hybrid = build_score_matrix(dets, bboxes, 0.5, 0.5, 100.0)
    assert m_hybrid[0][0] > 0
