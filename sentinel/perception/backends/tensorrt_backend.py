"""ONNX / TensorRT inference backend for edge deployment.

Supports:
- `.onnx` models via onnxruntime (CPU or GPU with CUDA EP)
- `.engine` models via TensorRT Python API (optional, Jetson only)

Usage:
  perception.backend: tensorrt
  perception.model_path: models/yolov8n.onnx

For TensorRT engine files:
  perception.model_path: models/yolov8n_fp16.engine
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from sentinel.common.types import BoundingBox, Detection

_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat",
}


def _load_coco_names():
    """Return COCO class names — same mapping used by ultralytics."""
    try:
        from ultralytics import YOLO
        tmp = YOLO("yolov8n.pt")
        names = tmp.names
        del tmp
        return names
    except Exception:
        return _COCO_NAMES


class OnnxBackend:
    """YOLO inference via ONNX Runtime (.onnx files)."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        classes: list[str] | None = None,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._classes = set(classes) if classes else None
        self._image_size = image_size
        self._model_path = model_path

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime is not installed. Install with: pip install onnxruntime (or onnxruntime-gpu)"
            )

        providers: list[str] = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info("Loading ONNX model: {} (providers={})", model_path, providers)
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        # Try to load class names from the ONNX metadata
        self._names = self._load_names()

    def _load_names(self) -> dict:
        meta = self._session.get_modelmeta()
        if meta.custom_metadata_map and "names" in meta.custom_metadata_map:
            import ast
            try:
                return ast.literal_eval(meta.custom_metadata_map["names"])
            except Exception:
                pass
        return _load_coco_names()

    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp_utc: str) -> list[Detection]:
        img = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: img})
        return self._postprocess(outputs, frame.shape, frame_id, timestamp_utc)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = frame.copy()
        h, w = frame.shape[:2]
        scale = min(self._image_size / w, self._image_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = np.ascontiguousarray(img)
        import cv2
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self._image_size - new_w
        pad_h = self._image_size - new_h
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[np.newaxis]

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        orig_shape: tuple[int, ...],
        frame_id: int,
        timestamp_utc: str,
    ) -> list[Detection]:
        preds = outputs[0]
        # YOLOv8 output shape: (1, num_detections, 4+num_classes)
        if preds.ndim == 3:
            preds = preds[0]

        orig_h, orig_w = orig_shape[:2]
        scale = min(self._image_size / orig_w, self._image_size / orig_h)
        pad_w = self._image_size - int(orig_w * scale)
        pad_h = self._image_size - int(orig_h * scale)

        detections: list[Detection] = []
        for det in preds:
            bbox = det[:4]
            class_scores = det[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if conf < self._confidence_threshold:
                continue

            class_name = str(self._names.get(class_id, f"class_{class_id}"))
            if self._classes and class_name not in self._classes:
                continue

            # Convert from center format to xyxy and unpad/unscale
            cx, cy, bw, bh = bbox
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale

            detections.append(
                Detection(
                    frame_id=frame_id,
                    timestamp_utc=timestamp_utc,
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox_xyxy=BoundingBox(
                        x1=max(int(x1), 0),
                        y1=max(int(y1), 0),
                        x2=min(int(x2), orig_w),
                        y2=min(int(y2), orig_h),
                    ),
                    source=f"onnx:{self._model_path}",
                )
            )
        return detections

    def close(self) -> None:
        pass


class TensorRTBackend:
    """YOLO inference via ONNX Runtime or TensorRT.

    Automatically selects OnnxBackend for .onnx files.
    For .engine files, falls back to onnxruntime with TensorRT execution
    provider if available, otherwise raises a clear error.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.onnx",
        confidence_threshold: float = 0.35,
        classes: list[str] | None = None,
        device: str = "cuda",
        image_size: int = 640,
    ) -> None:
        path = Path(model_path)

        if path.suffix == ".onnx":
            self._backend = OnnxBackend(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                classes=classes,
                device=device,
                image_size=image_size,
            )
        elif path.suffix == ".engine":
            self._backend = self._load_engine(model_path, confidence_threshold, classes, device, image_size)
        else:
            raise ValueError(f"Unsupported model format: {path.suffix} (expected .onnx or .engine)")

    def _load_engine(self, model_path, conf_thresh, classes, device, image_size):
        """Try to load TensorRT engine via onnxruntime with TrtEP or direct API."""
        try:
            import onnxruntime as ort
            providers = [
                ("TensorrtExecutionProvider", {"trt_engine_cache_enable": True}),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            logger.info("Loading TensorRT engine via onnxruntime: {}", model_path)
            session = ort.InferenceSession(model_path, providers=providers)
            # Wrap in OnnxBackend-style usage
            backend = OnnxBackend.__new__(OnnxBackend)
            backend._session = session
            backend._input_name = session.get_inputs()[0].name
            backend._confidence_threshold = conf_thresh
            backend._classes = set(classes) if classes else None
            backend._image_size = image_size
            backend._model_path = model_path
            backend._names = backend._load_names()
            return backend
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load TensorRT engine: {model_path}\n"
                f"Error: {exc}\n\n"
                f"Ensure TensorRT + onnxruntime-gpu are installed, or use .onnx format instead."
            ) from exc

    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp_utc: str) -> list[Detection]:
        return self._backend.detect_frame(frame, frame_id, timestamp_utc)

    def close(self) -> None:
        self._backend.close()
