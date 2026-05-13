"""Tests for TensorRT / ONNX backend."""

from __future__ import annotations

import pytest


def test_tensorrt_backend_invalid_extension():
    """Invalid file extension should raise ValueError."""
    from sentinel.perception.backends.tensorrt_backend import TensorRTBackend
    with pytest.raises(ValueError, match="Unsupported model format"):
        TensorRTBackend(model_path="model.txt")


def test_tensorrt_backend_engine_file_missing():
    """Missing .engine file should raise RuntimeError."""
    from sentinel.perception.backends.tensorrt_backend import TensorRTBackend
    with pytest.raises(RuntimeError, match="Cannot load TensorRT engine"):
        TensorRTBackend(model_path="nonexistent.engine")


def test_tensorrt_backend_onnx_missing():
    """Missing .onnx file should fail gracefully."""
    from sentinel.perception.backends.tensorrt_backend import TensorRTBackend
    with pytest.raises(Exception):
        TensorRTBackend(model_path="nonexistent.onnx")


def test_onnx_backend_skips_when_onnxruntime_missing(monkeypatch):
    """If onnxruntime is not available, OnnxBackend raises RuntimeError."""
    import sys
    onnx_mods = [m for m in sys.modules if m.startswith("onnxruntime")]
    saved = {m: sys.modules.pop(m) for m in onnx_mods}

    # Block import
    import importlib
    real_find = importlib.find_loader if hasattr(importlib, "find_loader") else None

    class FakeOnnxBackend:
        pass

    try:
        # The import check happens in __init__
        with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
            from sentinel.perception.backends.tensorrt_backend import OnnxBackend
            OnnxBackend(model_path="test.onnx")
    finally:
        for m, mod in saved.items():
            sys.modules[m] = mod
