"""Hybrid association scoring for detection-to-track matching."""

from __future__ import annotations

from sentinel.common.types import BoundingBox, Detection

from .iou import bbox_iou
from .motion import center_distance


def compute_hybrid_score(
    iou: float,
    center_dist: float,
    distance_threshold: float,
    iou_weight: float,
    distance_weight: float,
) -> float:
    if distance_threshold > 0:
        normalized_dist = 1.0 - min(center_dist / distance_threshold, 1.0)
    else:
        normalized_dist = 0.0
    return iou_weight * iou + distance_weight * normalized_dist


def build_score_matrix(
    detections: list[Detection],
    candidate_bboxes: list[BoundingBox],
    iou_weight: float,
    distance_weight: float,
    distance_threshold: float,
    class_names: list[str] | None = None,
    detection_classes: list[str] | None = None,
) -> list[list[float]]:
    rows = len(detections)
    cols = len(candidate_bboxes)
    matrix: list[list[float]] = []
    for di in range(rows):
        row: list[float] = []
        for ci in range(cols):
            if class_names and detection_classes:
                if detection_classes[di] != class_names[ci]:
                    row.append(0.0)
                    continue
            iou = bbox_iou(detections[di].bbox_xyxy, candidate_bboxes[ci])
            if distance_weight > 0 and distance_threshold > 0:
                dist = center_distance(detections[di].bbox_xyxy, candidate_bboxes[ci])
                score = compute_hybrid_score(iou, dist, distance_threshold, iou_weight, distance_weight)
            else:
                score = iou * iou_weight
            row.append(score)
        matrix.append(row)
    return matrix
