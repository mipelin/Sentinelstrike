"""Perception runner — orchestrates video source, backend, events, and output."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.time import utc_now_iso
from sentinel.common.types import Detection

from .backends import PerceptionBackend
from .drawing import draw_detections
from .metrics import compute_perception_metrics
from .mock_backend import MockPerceptionBackend
from .video_source import VideoSource


def _create_backend(config) -> PerceptionBackend:
    if config.backend == "mock":
        return MockPerceptionBackend(classes=config.classes, confidence_threshold=config.confidence_threshold)
    if config.backend == "yolo":
        from .yolo_backend import YoloPerceptionBackend

        return YoloPerceptionBackend(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            classes=config.classes,
            device=config.device,
            image_size=config.image_size,
        )
    if config.backend == "tensorrt":
        from .backends.tensorrt_backend import TensorRTBackend

        return TensorRTBackend(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            classes=config.classes,
            device=config.device,
            image_size=config.image_size,
        )
    raise ValueError(f"Unsupported perception backend: {config.backend}")


class PerceptionRunner:
    def __init__(
        self,
        config,
        mission_id: str,
        event_bus: EventBus | None = None,
        run_dir: Path | None = None,
        tracker_runner=None,
    ) -> None:
        self._config = config
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._run_dir = run_dir
        self._tracker_runner = tracker_runner

    def _publish(self, event_type: str, **payload: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="perception",
                    event_type=event_type,
                    payload=payload,
                )
            )

    def run(self) -> dict:
        started_at = utc_now_iso()
        self._publish("perception_started", source=self._config.source, backend=self._config.backend)
        logger.info("Perception started: backend={}, source={}", self._config.backend, self._config.source)

        backend = _create_backend(self._config)
        source = VideoSource(
            source=self._config.source,
            frame_stride=self._config.frame_stride,
            max_frames=self._config.max_frames,
        )

        detections_file = None
        if self._run_dir and self._config.save_detections_jsonl:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            detections_file = (self._run_dir / "detections.jsonl").open("w", encoding="utf-8")

        video_writer = None
        all_detections: list[Detection] = []
        frame_count = 0

        try:
            for frame_data in source:
                frame: np.ndarray = frame_data["frame"]
                fid = frame_data["frame_id"]
                ts = frame_data["timestamp_utc"]
                w, h = frame_data["width"], frame_data["height"]

                dets = backend.detect_frame(frame, fid, ts)
                frame_count += 1
                all_detections.extend(dets)

                self._publish(
                    "perception_frame_processed",
                    frame_id=fid,
                    detection_count=len(dets),
                )

                for det in dets:
                    self._publish(
                        "perception_detection",
                        frame_id=det.frame_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox_xyxy=[det.bbox_xyxy.x1, det.bbox_xyxy.y1, det.bbox_xyxy.x2, det.bbox_xyxy.y2],
                    )
                    if detections_file:
                        detections_file.write(det.model_dump_json() + "\n")

                if self._tracker_runner is not None:
                    self._tracker_runner.process_frame_detections(fid, ts, dets)

                if self._config.output_annotated_video and self._run_dir:
                    if video_writer is None:
                        self._run_dir.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                        video_writer = cv2.VideoWriter(str(self._run_dir / "annotated.mp4"), fourcc, 20.0, (w, h))
                    annotated = draw_detections(frame, dets)
                    video_writer.write(annotated)
        finally:
            backend.close()
            source.close()
            if detections_file:
                detections_file.close()
            if video_writer:
                video_writer.release()

        ended_at = utc_now_iso()
        metrics = compute_perception_metrics(frame_count, all_detections, started_at, ended_at)

        if self._run_dir:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            (self._run_dir / "perception_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        self._publish("perception_completed", **metrics)
        logger.info(
            "Perception completed: {} frames, {} detections",
            metrics["frame_count"],
            metrics["detection_count"],
        )

        if self._tracker_runner is not None:
            self._tracker_runner.close(frame_count=frame_count)

        return metrics
