"""Motion estimation and bbox prediction for tracking."""

from __future__ import annotations

import math

from sentinel.common.types import BoundingBox

from .iou import bbox_center


def estimate_velocity(prev_bbox: BoundingBox, curr_bbox: BoundingBox) -> tuple[float, float]:
    pc = bbox_center(prev_bbox)
    cc = bbox_center(curr_bbox)
    return (cc[0] - pc[0], cc[1] - pc[1])


def predict_bbox(last_bbox: BoundingBox, velocity: tuple[float, float], frames: int) -> BoundingBox:
    cx, cy = bbox_center(last_bbox)
    dx = velocity[0] * frames
    dy = velocity[1] * frames
    hw = (last_bbox.x2 - last_bbox.x1) / 2.0
    hh = (last_bbox.y2 - last_bbox.y1) / 2.0
    ncx = cx + dx
    ncy = cy + dy
    raw_x1 = max(int(round(ncx - hw)), 0)
    raw_y1 = max(int(round(ncy - hh)), 0)
    raw_x2 = max(int(round(ncx + hw)), raw_x1 + 1)
    raw_y2 = max(int(round(ncy + hh)), raw_y1 + 1)
    return BoundingBox(x1=raw_x1, y1=raw_y1, x2=raw_x2, y2=raw_y2)


def center_distance(a: BoundingBox, b: BoundingBox) -> float:
    ca = bbox_center(a)
    cb = bbox_center(b)
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
