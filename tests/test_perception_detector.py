"""Tests for PerceptionRunner with mock backend."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.common.event_bus import EventBus
from sentinel.perception.detector import PerceptionRunner


def _create_video(path: Path, frames: int = 30, w=320, h=240) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (w, h))
    for _ in range(frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(f)
    writer.release()
    return path


def _config(tmp_path, source: str, **overrides):
    from sentinel.config.schema import PerceptionConfig

    defaults = dict(
        enabled=True,
        backend="mock",
        model_path="yolov8n.pt",
        source=source,
        output_annotated_video=False,
        confidence_threshold=0.35,
        frame_stride=1,
        max_frames=30,
        classes=["person", "car"],
        save_detections_jsonl=True,
    )
    defaults.update(overrides)
    return PerceptionConfig(**defaults)


def test_runner_creates_detections_jsonl(tmp_path):
    vid = _create_video(tmp_path / "vid.mp4", frames=20)
    cfg = _config(tmp_path, source=str(vid))
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner = PerceptionRunner(config=cfg, mission_id="test", run_dir=run_dir)
    metrics = runner.run()

    det_file = run_dir / "detections.jsonl"
    assert det_file.exists()
    lines = det_file.read_text().strip().split("\n")
    assert len(lines) == metrics["detection_count"]
    for line in lines:
        d = json.loads(line)
        assert "class_name" in d


def test_runner_creates_metrics_json(tmp_path):
    vid = _create_video(tmp_path / "vid.mp4", frames=15)
    cfg = _config(tmp_path, source=str(vid))
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner = PerceptionRunner(config=cfg, mission_id="test", run_dir=run_dir)
    runner.run()

    metrics_file = run_dir / "perception_metrics.json"
    assert metrics_file.exists()
    m = json.loads(metrics_file.read_text())
    assert "frame_count" in m
    assert "detection_count" in m
    assert "classes" in m


def test_runner_publishes_events(tmp_path):
    vid = _create_video(tmp_path / "vid.mp4", frames=15)
    cfg = _config(tmp_path, source=str(vid))
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)

    runner = PerceptionRunner(config=cfg, mission_id="test", event_bus=bus)
    runner.run()

    types = [e.event_type for e in events]
    assert "perception_started" in types
    assert "perception_completed" in types
    assert "perception_frame_processed" in types


def test_invalid_backend_raises(tmp_path):

    with pytest.raises(ValueError):
        from sentinel.perception.detector import _create_backend

        _create_backend(_config(tmp_path, source="x", backend="invalid"))
