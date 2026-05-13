"""Tests for RealTimeMissionLoop in mock mode."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.common.event_bus import EventBus
from sentinel.common.types import BoundingBox, Detection, GeoObservation, Track
from sentinel.config.schema import AppConfig
from sentinel.realtime.loop import RealTimeMissionLoop


def _make_video(tmp_path: Path, n_frames: int = 25) -> str:
    path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (320, 240))
    for i in range(n_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 80), (220, 160), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def _config() -> AppConfig:
    return AppConfig(
        system={"vehicle_id": "uav_test"},
        tracker={"enabled": True},
        geolocalizer={"enabled": True, "camera": {"pitch_deg": -45.0}},
        tak={"enabled": True, "mode": "dry_run"},
    )


class MockPerceptionBackend:
    def __init__(self, classes=None, confidence_threshold=0.5):
        self._classes = classes or ["person"]
        self._threshold = confidence_threshold

    def detect_frame(self, frame, frame_id, timestamp_utc):
        h, w = frame.shape[:2]
        if frame_id % 5 != 0:
            return []
        return [
            Detection(
                frame_id=frame_id,
                timestamp_utc=timestamp_utc,
                class_name=self._classes[0],
                confidence=0.9,
                bbox_xyxy=BoundingBox(x1=10, y1=10, x2=100, y2=100),
                source="mock_test",
            )
        ]

    def close(self):
        pass


class MockTrackerRunner:
    def __init__(self):
        self.frame_data: list = []

    def process_frame_detections(self, frame_id, timestamp_utc, detections):
        tracks = []
        for det in detections:
            tracks.append(
                Track(
                    track_id=f"T-{frame_id}-{det.detection_id or '0'}",
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                    last_seen_utc=timestamp_utc,
                    first_seen_utc=timestamp_utc,
                )
            )
        self.frame_data.append((frame_id, len(detections)))
        return tracks

    def close(self, frame_count=None):
        pass

    def get_all_tracks(self):
        return []


class MockGeoRunner:
    def __init__(self):
        self.calls = 0

    def process_tracks(self, tracks, vehicle_state, timestamp_utc=None):
        self.calls += 1
        observations = []
        for trk in tracks:
            observations.append(
                GeoObservation(
                    observation_id=f"OBS-{trk.track_id}",
                    mission_id="test",
                    track_id=trk.track_id,
                    class_name=trk.class_name,
                    timestamp_utc=timestamp_utc or "2026-01-01T00:00:00.000000Z",
                    estimated_location={"lat": 38.0, "lon": -8.0, "alt_m": 0.0},
                    confidence=trk.confidence,
                )
            )
        return observations

    def close(self):
        pass


class MockTakBridge:
    def __init__(self):
        self.messages_sent = 0

    def send_geo_observation(self, obs):
        self.messages_sent += 1
        return True

    def send_vehicle_state(self, state):
        self.messages_sent += 1
        return True

    def close(self):
        return {"message_count": self.messages_sent}


def _run_loop(tmp_path, max_frames=25, target_fps=200.0, **kwargs):
    video_path = _make_video(tmp_path)
    cfg = _config()
    event_bus = EventBus()
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id="test_loop",
        event_bus=event_bus,
        run_dir=run_dir,
        perception_backend=MockPerceptionBackend(),
        tracker_runner=MockTrackerRunner(),
        geolocalization_runner=MockGeoRunner(),
        tak_bridge=MockTakBridge(),
        max_frames=max_frames,
        target_fps=target_fps,
        video_source=video_path,
        **kwargs,
    )
    return loop.run(), run_dir


def test_loop_returns_metrics(tmp_path):
    metrics, _ = _run_loop(tmp_path)
    assert metrics["frame_count"] == 25
    assert metrics["average_loop_hz"] >= 0
    assert metrics["average_frame_processing_ms"] >= 0


def test_loop_writes_realtime_metrics_json(tmp_path):
    _, run_dir = _run_loop(tmp_path)
    metrics_path = run_dir / "realtime_metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert "frame_count" in data
    assert "average_loop_hz" in data


def test_loop_writes_detections_jsonl(tmp_path):
    _, run_dir = _run_loop(tmp_path)
    det_path = run_dir / "detections.jsonl"
    assert det_path.exists()
    lines = det_path.read_text().strip().splitlines()
    assert len(lines) > 0


def test_loop_writes_tracks_jsonl(tmp_path):
    _, run_dir = _run_loop(tmp_path)
    trk_path = run_dir / "tracks.jsonl"
    assert trk_path.exists()
    lines = trk_path.read_text().strip().splitlines()
    assert len(lines) > 0


def test_loop_writes_geo_observations_jsonl(tmp_path):
    _, run_dir = _run_loop(tmp_path)
    geo_path = run_dir / "geo_observations.jsonl"
    assert geo_path.exists()
    lines = geo_path.read_text().strip().splitlines()
    assert len(lines) > 0


def test_loop_writes_tak_messages_jsonl(tmp_path):
    _, run_dir = _run_loop(tmp_path)
    tak_path = run_dir / "tak_messages.jsonl"
    assert tak_path.exists()
    lines = tak_path.read_text().strip().splitlines()
    assert len(lines) > 0


def test_loop_respects_max_frames(tmp_path):
    metrics, _ = _run_loop(tmp_path, max_frames=10)
    assert metrics["frame_count"] == 10


def test_loop_detection_count(tmp_path):
    metrics, _ = _run_loop(tmp_path, max_frames=25)
    # MockBackend returns detection on frame_id % 5 == 0: frames 0,5,10,15,20 = 5 detections
    assert metrics["detection_count"] == 5


def test_loop_static_telemetry(tmp_path):
    metrics, _ = _run_loop(tmp_path)
    assert metrics["telemetry_stale_count"] == 0


def test_loop_writes_annotated_frames(tmp_path):
    _, run_dir = _run_loop(tmp_path, max_frames=6)
    frames_dir = run_dir / "frames"
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("frame_*.jpg"))) == 6
    assert (run_dir / "latest.jpg").exists()


def test_loop_updates_tracker_every_frame(tmp_path):
    video_path = _make_video(tmp_path, n_frames=8)
    cfg = _config()
    event_bus = EventBus()
    run_dir = tmp_path / "runs" / "tracker_every_frame"
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = MockTrackerRunner()

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id="tracker_every_frame",
        event_bus=event_bus,
        run_dir=run_dir,
        perception_backend=MockPerceptionBackend(),
        tracker_runner=tracker,
        geolocalization_runner=MockGeoRunner(),
        tak_bridge=MockTakBridge(),
        max_frames=8,
        target_fps=200.0,
        video_source=video_path,
    )
    loop.run()
    assert len(tracker.frame_data) == 8


def test_loop_publishes_events(tmp_path):
    video_path = _make_video(tmp_path)
    cfg = _config()
    event_bus = EventBus()
    run_dir = tmp_path / "runs" / "evt_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    collected_events = []
    event_bus.subscribe("*", lambda e: collected_events.append(e))

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id="evt_test",
        event_bus=event_bus,
        run_dir=run_dir,
        perception_backend=MockPerceptionBackend(),
        tracker_runner=MockTrackerRunner(),
        geolocalization_runner=MockGeoRunner(),
        tak_bridge=MockTakBridge(),
        max_frames=10,
        target_fps=200.0,
        video_source=video_path,
    )
    loop.run()

    event_types = {e.event_type for e in collected_events}
    assert "realtime_loop_started" in event_types
    assert "realtime_loop_completed" in event_types
    assert "realtime_frame_processed" in event_types
    assert "realtime_detection" in event_types


def test_loop_no_perception_still_works(tmp_path):
    video_path = _make_video(tmp_path)
    cfg = _config()
    event_bus = EventBus()
    run_dir = tmp_path / "runs" / "no_perc"
    run_dir.mkdir(parents=True, exist_ok=True)

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id="no_perc",
        event_bus=event_bus,
        run_dir=run_dir,
        max_frames=10,
        target_fps=200.0,
        video_source=video_path,
    )
    metrics = loop.run()
    assert metrics["frame_count"] == 10
    assert metrics["detection_count"] == 0


def test_loop_invalid_video_source(tmp_path):
    cfg = _config()
    event_bus = EventBus()
    run_dir = tmp_path / "runs" / "bad"
    run_dir.mkdir(parents=True, exist_ok=True)

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id="bad",
        event_bus=event_bus,
        run_dir=run_dir,
        max_frames=10,
        target_fps=200.0,
        video_source="/nonexistent/video.mp4",
    )
    with pytest.raises(RuntimeError, match="Cannot open video source"):
        loop.run()
