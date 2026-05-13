"""Tests for TrackingRunner."""

from sentinel.common.event_bus import EventBus
from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, Detection
from sentinel.config.schema import TrackerConfig
from sentinel.tracker.runner import TrackingRunner


def _det(frame_id: int = 0, x1=0, y1=0, x2=10, y2=10):
    return Detection(
        frame_id=frame_id,
        timestamp_utc=utc_now_iso(),
        class_name="person",
        confidence=0.8,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_runner_creates_tracks_jsonl(tmp_path):
    cfg = TrackerConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = TrackingRunner(config=cfg, mission_id="t", run_dir=run_dir)
    runner.process_frame_detections(0, utc_now_iso(), [_det()])
    runner.close(frame_count=1)
    assert (run_dir / "tracks.jsonl").exists()


def test_runner_creates_metrics_json(tmp_path):
    cfg = TrackerConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = TrackingRunner(config=cfg, mission_id="t", run_dir=run_dir)
    runner.process_frame_detections(0, utc_now_iso(), [_det()])
    runner.close(frame_count=1)
    assert (run_dir / "tracker_metrics.json").exists()


def test_runner_publishes_events():
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)
    cfg = TrackerConfig()
    runner = TrackingRunner(config=cfg, mission_id="t", event_bus=bus)
    runner.process_frame_detections(0, utc_now_iso(), [_det()])
    runner.close(frame_count=1)
    types = [e.event_type for e in events]
    assert "tracker_updated" in types
    assert "track_created" in types
    assert "tracking_completed" in types


def test_perception_with_tracking_integration(tmp_path):
    """PerceptionRunner + TrackingRunner end-to-end with mock backend."""
    import cv2
    import numpy as np

    vid_path = tmp_path / "vid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(vid_path), fourcc, 20.0, (320, 240))
    for _ in range(15):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    from sentinel.config.schema import PerceptionConfig
    from sentinel.perception.detector import PerceptionRunner

    perc_cfg = PerceptionConfig(source=str(vid_path), backend="mock", max_frames=15, classes=["person"])
    track_cfg = TrackerConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)

    tracker_runner = TrackingRunner(config=track_cfg, mission_id="t", event_bus=bus, run_dir=run_dir)

    perc_runner = PerceptionRunner(
        config=perc_cfg,
        mission_id="t",
        event_bus=bus,
        run_dir=run_dir,
        tracker_runner=tracker_runner,
    )
    perc_runner.run()

    assert (run_dir / "tracks.jsonl").exists()
    assert (run_dir / "tracker_metrics.json").exists()
    types = [e.event_type for e in events]
    assert "track_created" in types
    assert "tracking_completed" in types


def test_tracks_jsonl_lost_status_has_positive_lost_frames(tmp_path):
    cfg = TrackerConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = TrackingRunner(config=cfg, mission_id="t", run_dir=run_dir)

    # Frame 0: create track with detection
    runner.process_frame_detections(0, utc_now_iso(), [_det()])
    # Frame 1: no detection → track becomes lost
    runner.process_frame_detections(1, utc_now_iso(), [])
    runner.close(frame_count=2)

    lines = (run_dir / "tracks.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 2

    from sentinel.common.types import Track

    for line in lines:
        trk = Track.model_validate_json(line)
        if trk.status == "lost":
            assert trk.lost_frames >= 1, f"lost track {trk.track_id} has lost_frames={trk.lost_frames}"
        if trk.status == "active":
            assert trk.lost_frames == 0, f"active track {trk.track_id} has lost_frames={trk.lost_frames}"
