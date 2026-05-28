"""PerceptionWorker — independent YOLO inference on the latest frame.

Consumes the newest frame from a LatestSlot[np.ndarray] (written by
CameraWorker), runs YOLO or mock inference, and publishes DetectionSet
into an output LatestSlot. Never accumulates backlog — always processes
only the latest available frame.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import DetectionSet, FrameSnapshot
from sentinel.runtime.worker import Worker

_CLASS_ALIASES = {
    "walker": "person",
    "pedestrian": "person",
    "human": "person",
    "vehicle": "car",
    "automobile": "car",
    "auto": "car",
}


def _normalize_class_name(name: str) -> str:
    return _CLASS_ALIASES.get(name, name)


class PerceptionWorker(Worker):
    """YOLO inference worker.

    Reads the latest frame from ``frame_slot``, runs inference, and publishes
    a DetectionSet into ``detection_slot``. Skips frames that haven't changed
    since the last inference (no backlog, no FIFO).

    Usage::

        perc = PerceptionWorker(
            frame_slot=camera.frame_slot,
            meta_slot=camera.meta_slot,
            backend="yolo",
            model_path="yolov8n.pt",
            device="cuda",
        )
        perc.load_model()
        perc.start()
        ...
        det_set, seq, ts = perc.detection_slot.read()
        perc.stop()
    """

    def __init__(
        self,
        frame_slot: LatestSlot[np.ndarray],
        meta_slot: LatestSlot[FrameSnapshot],
        *,
        backend: str = "yolo",
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence: float = 0.3,
        classes: list[str] | None = None,
        imgsz: int | None = None,
        name: str = "perception",
        hz: float = 15.0,
    ) -> None:
        super().__init__(name=name, hz=hz)
        self._frame_slot = frame_slot
        self._meta_slot = meta_slot
        self._backend = backend
        self._model_path = model_path
        self._device = self._resolve_device(device)
        self._confidence = confidence
        self._classes = classes or ["person", "car", "truck", "bus"]
        self._imgsz = imgsz

        # Public output
        self.detection_slot: LatestSlot[DetectionSet] = LatestSlot()

        # Internal state
        self._model: Any = None
        self._last_frame_seq: int = 0

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def load_model(self) -> None:
        """Load YOLO model. Must be called before start()."""
        if self._backend != "yolo":
            return
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)
        logger.info(
            "PerceptionWorker loaded {} (device={})",
            self._model_path,
            self._device,
        )

    def tick(self) -> None:
        """Read latest frame, run inference, publish DetectionSet."""
        # Read latest frame
        frame, frame_seq, frame_ts = self._frame_slot.read()

        if frame is None:
            self._metrics.record_skip()
            return

        # Skip if we already processed this frame (no backlog)
        if frame_seq == self._last_frame_seq:
            self._metrics.record_skip()
            return

        self._last_frame_seq = frame_seq

        # Read frame metadata for frame_id
        meta_val, _, _ = self._meta_slot.read()
        frame_id = meta_val.frame_id if meta_val is not None else 0

        # Run inference
        t0 = time.monotonic()
        detections, inference_ms = self._run_inference(frame)

        # Publish DetectionSet
        det_set = DetectionSet(
            frame_id=frame_id,
            seq=self._last_frame_seq,
            ts_monotonic=time.monotonic(),
            detections=detections,
            backend=self._backend,
            inference_ms=inference_ms,
        )
        self.detection_slot.write(det_set)

    def _run_inference(
        self, frame: np.ndarray,
    ) -> tuple[list[dict], float]:
        """Run detection backend. Returns (detections, inference_ms)."""
        if self._backend == "mock":
            return self._mock_inference(frame), 0.0

        if self._model is None:
            return [], 0.0

        t0 = time.monotonic()
        detections = self._yolo_inference(frame)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return detections, elapsed_ms

    def _yolo_inference(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO detection and extract results."""
        kwargs: dict = {
            "conf": self._confidence,
            "verbose": False,
            "device": self._device,
        }
        if self._imgsz is not None:
            kwargs["imgsz"] = self._imgsz
        results = self._model(frame, **kwargs)
        return self._extract_detections(results)

    def _extract_detections(self, results: list) -> list[dict]:
        """Parse YOLO results into detection dicts."""
        detections: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw_cls_name = r.names[cls_id]
                cls_name = _normalize_class_name(raw_cls_name)
                if self._classes and cls_name not in self._classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                track_id = None
                if hasattr(box, "id") and box.id is not None:
                    track_id = int(box.id[0])
                detections.append({
                    "class": cls_name,
                    "raw_class": raw_cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                })
        return detections

    @staticmethod
    def _mock_inference(frame: np.ndarray) -> list[dict]:
        """Generate mock detections for testing."""
        h, w = frame.shape[:2]
        return [
            {
                "class": "person",
                "confidence": 0.85,
                "bbox": [w * 0.3, h * 0.3, w * 0.7, h * 0.8],
                "track_id": None,
            },
            {
                "class": "car",
                "confidence": 0.72,
                "bbox": [w * 0.05, h * 0.5, w * 0.4, h * 0.9],
                "track_id": None,
            },
        ]
