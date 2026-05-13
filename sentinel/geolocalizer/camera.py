"""Camera model and pixel-to-ray conversion (pinhole approximation).

Approximations:
- No lens distortion correction.
- Linear FOV mapping (tan model would be more accurate at edges).
- Principal point = image center.
- No intrinsic matrix — just FOV-based mapping.
"""

from __future__ import annotations

import math

from sentinel.common.types import BoundingBox, CameraModel


def bbox_center_px(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def pixel_to_normalized_camera_ray(
    x_px: float,
    y_px: float,
    camera: CameraModel,
) -> tuple[float, float, float]:
    """Convert pixel coords to a unit-ish ray in camera coordinates.

    Camera coordinate convention:
      +x = right
      +y = down
      +z = forward (along optical axis)

    Returns (rx, ry, rz) — not necessarily unit length, but proportional
    to the direction from the camera center through the pixel.
    """
    cx = camera.width_px / 2.0
    cy = camera.height_px / 2.0

    # Pixel offset from center, normalized to [-0.5, 0.5]
    dx = (x_px - cx) / camera.width_px
    dy = (y_px - cy) / camera.height_px

    # Map to angle using half-FOV
    half_hfov_rad = math.radians(camera.horizontal_fov_deg / 2.0)
    half_vfov_rad = math.radians(camera.vertical_fov_deg / 2.0)

    # Forward component dominates; lateral proportional to offset
    rz = 1.0
    rx = dx * math.tan(half_hfov_rad)
    ry = dy * math.tan(half_vfov_rad)

    return (rx, ry, rz)
