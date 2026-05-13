"""Tests for motion estimation and prediction."""

from sentinel.common.types import BoundingBox
from sentinel.tracker.motion import center_distance, estimate_velocity, predict_bbox


def test_estimate_velocity_zero():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    assert estimate_velocity(a, a) == (0.0, 0.0)


def test_estimate_velocity_positive():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=10, y1=5, x2=20, y2=15)
    vx, vy = estimate_velocity(a, b)
    assert vx == 10.0  # center moves from 5 to 15
    assert vy == 5.0   # center moves from 5 to 10


def test_estimate_velocity_negative():
    a = BoundingBox(x1=10, y1=10, x2=20, y2=20)
    b = BoundingBox(x1=0, y1=5, x2=10, y2=15)
    vx, vy = estimate_velocity(a, b)
    assert vx == -10.0
    assert vy == -5.0


def test_predict_bbox_one_frame():
    bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
    predicted = predict_bbox(bbox, (10.0, 5.0), 1)
    # Center (150, 150) → (160, 155), hw=50, hh=50
    assert predicted.x1 == 110
    assert predicted.y1 == 105
    assert predicted.x2 == 210
    assert predicted.y2 == 205


def test_predict_bbox_multiple_frames():
    bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
    predicted = predict_bbox(bbox, (10.0, 0.0), 5)
    # Center 150 → 200, hw=50
    assert predicted.x1 == 150
    assert predicted.x2 == 250


def test_predict_bbox_zero_velocity():
    bbox = BoundingBox(x1=50, y1=50, x2=100, y2=100)
    predicted = predict_bbox(bbox, (0.0, 0.0), 3)
    assert predicted.x1 == 50
    assert predicted.y1 == 50
    assert predicted.x2 == 100
    assert predicted.y2 == 100


def test_predict_bbox_no_negative_coords():
    bbox = BoundingBox(x1=2, y1=2, x2=10, y2=10)
    predicted = predict_bbox(bbox, (-5.0, -5.0), 5)
    assert predicted.x1 >= 0
    assert predicted.y1 >= 0


def test_center_distance_same():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    assert center_distance(a, a) == 0.0


def test_center_distance_known():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)      # center (5, 5)
    b = BoundingBox(x1=10, y1=10, x2=20, y2=20)     # center (15, 15)
    d = center_distance(a, b)
    expected = (10.0 ** 2 + 10.0 ** 2) ** 0.5
    assert abs(d - expected) < 0.01


def test_center_distance_symmetric():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=100, y1=100, x2=110, y2=110)
    assert abs(center_distance(a, b) - center_distance(b, a)) < 0.001
