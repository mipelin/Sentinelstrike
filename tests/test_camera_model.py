"""Tests for camera model."""

import pytest

from sentinel.common.types import BoundingBox, CameraModel
from sentinel.geolocalizer.camera import bbox_center_px, pixel_to_normalized_camera_ray


def test_camera_model_valid():
    cam = CameraModel()
    assert cam.width_px == 640
    assert cam.height_px == 480
    assert cam.horizontal_fov_deg == 70.0


def test_camera_model_invalid_fov():
    with pytest.raises(ValueError):
        CameraModel(horizontal_fov_deg=0)
    with pytest.raises(ValueError):
        CameraModel(horizontal_fov_deg=180)
    with pytest.raises(ValueError):
        CameraModel(vertical_fov_deg=-5)


def test_bbox_center_px():
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120)
    cx, cy = bbox_center_px(bbox)
    assert cx == 60.0
    assert cy == 70.0


def test_pixel_to_normalized_camera_ray():
    cam = CameraModel()
    # Center pixel should give (0, 0, 1) direction
    rx, ry, rz = pixel_to_normalized_camera_ray(320, 240, cam)
    assert abs(rx) < 1e-9
    assert abs(ry) < 1e-9
    assert rz == 1.0


def test_pixel_off_center():
    cam = CameraModel()
    rx, ry, rz = pixel_to_normalized_camera_ray(640, 240, cam)
    assert rx > 0
    assert abs(ry) < 1e-9
    assert rz == 1.0
