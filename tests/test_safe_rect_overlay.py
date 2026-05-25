"""Tests for _safe_rect and overlay robustness in run_gazebo_yolo_test."""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _mock_gz(monkeypatch):
    mock_gz = types.ModuleType("gz")
    mock_transport = types.ModuleType("gz.transport13")
    mock_msgs = types.ModuleType("gz.msgs10")
    mock_image = types.ModuleType("gz.msgs10.image_pb2")

    class _MockNode:
        def subscribe(self, **kwargs):
            pass

    mock_transport.Node = _MockNode
    mock_image.Image = MagicMock

    mock_gz.transport13 = mock_transport
    mock_msgs.image_pb2 = mock_image
    mock_gz.msgs10 = mock_msgs

    monkeypatch.setitem(sys.modules, "gz", mock_gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", mock_transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", mock_msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.image_pb2", mock_image)


from apps.tools.run_gazebo_yolo_test import (
    _draw_isr_overlay,
    _draw_overlay,
    _safe_rect,
)


def _blank_1080p():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# --- _safe_rect unit tests ---

class TestSafeRect:
    def test_basic_int_coords(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10, 20, 100, 50, (0, 255, 0)) is True

    def test_float_coords(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10.7, 20.3, 100.9, 50.1, (255, 0, 0)) is True

    def test_numpy_scalar_coords(self):
        img = _blank_1080p()
        assert _safe_rect(
            img, np.float64(10.5), np.int32(20),
            np.float32(100.0), np.int64(50),
            (0, 0, 255),
        ) is True

    def test_zero_width_returns_false(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10, 20, 10, 50, (255, 255, 255)) is False

    def test_negative_width_returns_false(self):
        img = _blank_1080p()
        assert _safe_rect(img, 100, 20, 10, 50, (255, 255, 255)) is False

    def test_zero_height_returns_false(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10, 20, 50, 20, (255, 255, 255)) is False

    def test_fully_outside_image(self):
        img = _blank_1080p()
        assert _safe_rect(img, 2000, 20, 3000, 50, (255, 255, 255)) is False
        assert _safe_rect(img, 10, 2000, 50, 3000, (255, 255, 255)) is False
        assert _safe_rect(img, -100, -100, -10, -10, (255, 255, 255)) is False

    def test_clamps_negative_coords(self):
        img = _blank_1080p()
        assert _safe_rect(img, -5, -5, 50, 50, (128, 128, 128)) is True

    def test_clamps_past_image_bounds(self):
        img = _blank_1080p()
        assert _safe_rect(img, 1900, 1000, 2000, 1100, (128, 128, 128)) is True

    def test_color_as_list(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10, 20, 100, 50, [0, 255, 0]) is True

    def test_color_as_numpy_array(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10, 20, 100, 50, np.array([0, 255, 0])) is True

    def test_explicit_thickness(self):
        img = _blank_1080p()
        assert _safe_rect(img, 10, 20, 100, 50, (0, 255, 0), thickness=2) is True

    def test_does_not_crash_on_any_input(self):
        img = _blank_1080p()
        _safe_rect(img, math.nan, 0, 100, 100, (0, 0, 0))
        _safe_rect(img, math.inf, 0, 100, 100, (0, 0, 0))
        _safe_rect(img, 0, -math.inf, 100, 100, (0, 0, 0))


# --- Overlay integration tests with edge-case detections ---

def _base_detections():
    return [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
    }]


def test_overlay_basic_detections():
    img = _blank_1080p()
    result = _draw_overlay(img, _base_detections(), 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_basic_detections():
    img = _blank_1080p()
    result = _draw_isr_overlay(img, _base_detections(), 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_overlay_float_bbox():
    img = _blank_1080p()
    dets = [{"class": "car", "confidence": 0.8, "bbox": [100.5, 200.3, 400.7, 600.9]}]
    result = _draw_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_overlay_numpy_scalar_bbox():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": np.float64(0.75),
        "bbox": [np.float32(50), np.int32(50), np.float64(200), np.int64(300)],
    }]
    result = _draw_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_hypothesis_nan_probability():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "OCCLUDED", "probability": float("nan")},
            {"type": "CONTINUED_PATH", "probability": 0.6},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_hypothesis_inf_probability():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "OCCLUDED", "probability": float("inf")},
            {"type": "STATIONARY", "probability": -float("inf")},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_hypothesis_zero_probability():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "OCCLUDED", "probability": 0.0},
            {"type": "STATIONARY", "probability": -0.1},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_hypothesis_tiny_probability():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "OCCLUDED", "probability": 1e-10},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_hypothesis_unknown_type():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "UNKNOWN_TYPE", "probability": 0.8},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_bbox_near_edge():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [1850, 1000, 1950, 1080],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "OCCLUDED", "probability": 0.7},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_negative_bbox_coords():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [-50, -20, 200, 200],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "CONTINUED_PATH", "probability": 0.9},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_behavior_and_priority():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "behavior_pattern": "LOITERING",
        "priority_score": 0.75,
        "anomaly_score": 0.8,
        "concealment_affinity": 0.5,
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_lost_track_with_prediction():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "LOST",
        "predicted_position": [250.5, 350.3],
        "hypotheses": [
            {"type": "OCCLUDED", "probability": 0.6},
            {"type": "EXITED_FOV", "probability": 0.3},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_velocity_and_speed():
    img = _blank_1080p()
    dets = [{
        "class": "car",
        "confidence": 0.85,
        "bbox": [200, 200, 500, 400],
        "track_id": "TGT-002",
        "status": "CONFIRMED",
        "velocity": [3.5, -2.1],
        "speed": 15.0,
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_merge_group():
    img = _blank_1080p()
    dets = [
        {
            "class": "person",
            "confidence": 0.9,
            "bbox": [100, 100, 300, 400],
            "track_id": "TGT-001",
            "status": "CONFIRMED",
            "merge_group": ["TGT-002"],
        },
        {
            "class": "person",
            "confidence": 0.8,
            "bbox": [350, 100, 550, 400],
            "track_id": "TGT-002",
            "status": "CONFIRMED",
        },
    ]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_empty_detections():
    img = _blank_1080p()
    result = _draw_isr_overlay(img, [], 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_overlay_empty_detections():
    img = _blank_1080p()
    result = _draw_overlay(img, [], 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_occlusion_state():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "occlusion_state": "OCCLUDED",
        "occluded_frames": 15,
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_world_position():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "world_position": {"lat": 37.7749, "lon": -122.4194, "ground_distance_m": 25.0},
        "world_velocity": {"speed_mps": 1.5, "heading_deg": 90.0},
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape


def test_isr_overlay_many_hypotheses():
    img = _blank_1080p()
    dets = [{
        "class": "person",
        "confidence": 0.9,
        "bbox": [100, 100, 300, 400],
        "track_id": "TGT-001",
        "status": "CONFIRMED",
        "hypotheses": [
            {"type": "OCCLUDED", "probability": 0.4},
            {"type": "CONTINUED_PATH", "probability": 0.3},
            {"type": "EXITED_FOV", "probability": 0.2},
            {"type": "MERGED_GROUP", "probability": 0.1},
        ],
    }]
    result = _draw_isr_overlay(img, dets, 30.0, 25.0, 10.0, "cpu", 1)
    assert result.shape == img.shape
