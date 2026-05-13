"""YOLO perception backend — requires ultralytics optional dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from sentinel.common.types import BoundingBox, Detection


def _ensure_model(model_path: str) -> str:
    """Download model if it doesn't exist. Supports ultralytics model names."""
    if Path(model_path).exists():
        return model_path
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    local = models_dir / Path(model_path).name
    if local.exists():
        return str(local)
    logger.info("Model {} not found — ultralytics will auto-download", model_path)
    return model_path


class YoloPerceptionBackend:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        classes: list[str] | None = None,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._classes = set(classes) if classes else None
        self._device = device
        self._image_size = image_size
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install with: pip install -e .[perception]") from exc

        resolved = _ensure_model(model_path)
        logger.info("Loading YOLO model: {} (device={}, imgsz={})", resolved, device, image_size)
        self._model = YOLO(resolved)

    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp_utc: str) -> list[Detection]:
        results = self._model(frame, verbose=False, device=self._device, imgsz=self._image_size)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                if self._classes and class_name not in self._classes:
                    continue
                conf = float(box.conf[0])
                if conf < self._confidence_threshold:
                    continue
                xyxy = box.xyxy[0]
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                elif not isinstance(xyxy, np.ndarray):
                    xyxy = np.array(xyxy)
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp_utc=timestamp_utc,
                        class_name=class_name,
                        confidence=round(conf, 4),
                        bbox_xyxy=BoundingBox(
                            x1=int(xyxy[0]),
                            y1=int(xyxy[1]),
                            x2=int(xyxy[2]),
                            y2=int(xyxy[3]),
                        ),
                        source=f"yolo:{self._model_path}",
                    )
                )
        return detections

    def close(self) -> None:
        pass


# Backwards-compatible alias
YOLOBackend = YoloPerceptionBackend
