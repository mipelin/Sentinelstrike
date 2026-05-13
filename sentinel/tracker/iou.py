"""IoU computation for bounding box association."""

from __future__ import annotations

from sentinel.common.types import BoundingBox


def bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def bbox_area(bbox: BoundingBox) -> float:
    return float((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter_w = max(ix2 - ix1, 0)
    inter_h = max(iy2 - iy1, 0)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0

    return inter / union
