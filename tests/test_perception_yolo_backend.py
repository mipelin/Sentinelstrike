"""Tests for YOLO perception backend.

These tests work without ultralytics installed by mocking the YOLO model.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentinel.common.time import utc_now_iso
from sentinel.perception.yolo_backend import _ensure_model

# ---------------------------------------------------------------------------
# _ensure_model helper (pure function, no ultralytics needed)
# ---------------------------------------------------------------------------


def test_ensure_model_returns_existing_path(tmp_path):
    model_file = tmp_path / "yolov8n.pt"
    model_file.write_text("fake model")
    assert _ensure_model(str(model_file)) == str(model_file)


def test_ensure_model_returns_original_if_not_found(tmp_path):
    result = _ensure_model("yolov8n.pt")
    assert result == "yolov8n.pt"


# ---------------------------------------------------------------------------
# YoloPerceptionBackend constructor without ultralytics
# ---------------------------------------------------------------------------


def test_constructor_fails_without_ultralytics():
    import importlib.util
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics is installed — skipping no-ultralytics test")
    with pytest.raises(RuntimeError, match="pip install"):
        from sentinel.perception.yolo_backend import YoloPerceptionBackend

        YoloPerceptionBackend(model_path="yolov8n.pt")


# ---------------------------------------------------------------------------
# Helpers for mocked YOLO tests
# ---------------------------------------------------------------------------


def _make_mock_box(cls_id=0, conf=0.87, xyxy=None):
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [np.array(xyxy or [50.0, 60.0, 200.0, 300.0])]
    return box


def _make_mock_model(boxes, names):
    result = MagicMock()
    result.boxes = boxes
    result.names = names
    return MagicMock(return_value=[result])


def _backend_with_mock(mock_model):
    """Create a YoloPerceptionBackend with a mocked YOLO, reloading the module."""
    mock_yolo_class = MagicMock(return_value=mock_model)
    fake_ultralytics = MagicMock(YOLO=mock_yolo_class)
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        import sentinel.perception.yolo_backend as mod

        importlib.reload(mod)
        backend = mod.YoloPerceptionBackend(model_path="yolov8n.pt", confidence_threshold=0.3, classes=["person"])
    return backend, mock_model


# ---------------------------------------------------------------------------
# detect_frame tests with mocked YOLO
# ---------------------------------------------------------------------------


def test_detect_frame_returns_detections():
    mock_model = _make_mock_model([_make_mock_box()], {0: "person"})
    backend, _ = _backend_with_mock(mock_model)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = backend.detect_frame(frame, frame_id=0, timestamp_utc=utc_now_iso())

    assert len(dets) == 1
    det = dets[0]
    assert det.class_name == "person"
    assert det.confidence >= 0.3
    assert det.bbox_xyxy.x1 == 50
    assert det.bbox_xyxy.y1 == 60
    assert det.bbox_xyxy.x2 == 200
    assert det.bbox_xyxy.y2 == 300
    assert det.source.startswith("yolo:")


def test_detect_frame_filters_by_class():
    mock_model = _make_mock_model([_make_mock_box()], {0: "dog"})
    backend, _ = _backend_with_mock(mock_model)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = backend.detect_frame(frame, frame_id=0, timestamp_utc=utc_now_iso())
    assert len(dets) == 0


def test_detect_frame_filters_by_confidence():
    mock_model = _make_mock_model([_make_mock_box(conf=0.2)], {0: "person"})

    mock_yolo_class = MagicMock(return_value=mock_model)
    fake_ultralytics = MagicMock(YOLO=mock_yolo_class)
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        import sentinel.perception.yolo_backend as mod

        importlib.reload(mod)
        backend = mod.YoloPerceptionBackend(model_path="yolov8n.pt", confidence_threshold=0.5)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = backend.detect_frame(frame, frame_id=0, timestamp_utc=utc_now_iso())
    assert len(dets) == 0


def test_detect_frame_passes_device_and_image_size():
    mock_model = _make_mock_model([_make_mock_box()], {0: "person"})

    mock_yolo_class = MagicMock(return_value=mock_model)
    fake_ultralytics = MagicMock(YOLO=mock_yolo_class)
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        import sentinel.perception.yolo_backend as mod

        importlib.reload(mod)
        backend = mod.YoloPerceptionBackend(model_path="yolov8n.pt", device="cuda", image_size=1280)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    backend.detect_frame(frame, frame_id=0, timestamp_utc=utc_now_iso())

    mock_model.assert_called_once()
    call_kwargs = mock_model.call_args
    assert call_kwargs[1]["device"] == "cuda"
    assert call_kwargs[1]["imgsz"] == 1280


def test_multiple_detections():
    boxes = [
        _make_mock_box(cls_id=0, conf=0.9, xyxy=[10, 10, 100, 100]),
        _make_mock_box(cls_id=1, conf=0.75, xyxy=[200, 200, 350, 400]),
    ]
    mock_model = _make_mock_model(boxes, {0: "person", 1: "car"})

    mock_yolo_class = MagicMock(return_value=mock_model)
    fake_ultralytics = MagicMock(YOLO=mock_yolo_class)
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        import sentinel.perception.yolo_backend as mod

        importlib.reload(mod)
        backend = mod.YoloPerceptionBackend(model_path="yolov8n.pt", classes=["person", "car"])

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = backend.detect_frame(frame, frame_id=5, timestamp_utc=utc_now_iso())

    assert len(dets) == 2
    assert dets[0].class_name == "person"
    assert dets[1].class_name == "car"


def test_close_is_noop():
    mock_model = _make_mock_model([], {})
    backend, _ = _backend_with_mock(mock_model)
    backend.close()  # should not raise
