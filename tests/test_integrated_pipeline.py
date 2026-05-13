"""Tests for IntegratedMissionPipeline."""

import json
from pathlib import Path

from sentinel.common.types import GeoPoint, MissionRequest
from sentinel.config.schema import AppConfig
from sentinel.pipeline.integrated_pipeline import IntegratedMissionPipeline


def _mission_request():
    return MissionRequest(
        mission_id="test_mission",
        launch_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=0),
        area_of_interest=[
            GeoPoint(lat=38.000, lon=-8.000, alt_m=80),
            GeoPoint(lat=38.000, lon=-7.998, alt_m=80),
            GeoPoint(lat=38.002, lon=-7.998, alt_m=80),
            GeoPoint(lat=38.002, lon=-8.000, alt_m=80),
        ],
        search_pattern="lawnmower",
    )


def _config(tmp_path):
    return AppConfig(
        recorder={"output_dir": str(tmp_path / "runs"), "enabled": True},
        perception={"backend": "mock", "max_frames": 15, "source": str(tmp_path / "vid.mp4"), "classes": ["person"]},
        tracker={"enabled": True},
        geolocalizer={"enabled": True, "camera": {"pitch_deg": -45.0}},
        tak={"enabled": True, "mode": "dry_run"},
    )


def _ensure_video(tmp_path):
    import cv2
    import numpy as np

    vid_path = tmp_path / "vid.mp4"
    if vid_path.exists():
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(vid_path), fourcc, 20.0, (320, 240))
    for _ in range(15):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()


def test_pipeline_creates_run_dir(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    req = _mission_request()
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=req)
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert run_dir.exists()


def test_pipeline_creates_summary_json(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "pipeline_summary.json").exists()


def test_pipeline_creates_report(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "report.md").exists()


def test_pipeline_creates_mission_plan(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "mission_plan.json").exists()


def test_pipeline_creates_detections_jsonl(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "detections.jsonl").exists()


def test_pipeline_creates_tracks_jsonl(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "tracks.jsonl").exists()


def test_pipeline_creates_geo_observations(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "geo_observations.jsonl").exists()


def test_pipeline_creates_tak_messages(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "tak_messages.jsonl").exists()


def test_pipeline_summary_has_detections(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    assert summary["perception"]["detection_count"] >= 1


def test_pipeline_summary_has_tak_messages(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    assert summary["tak"]["message_count"] >= 1


def test_pipeline_events_contain_started_and_completed(tmp_path):
    _ensure_video(tmp_path)
    cfg = _config(tmp_path)
    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=_mission_request())
    summary = pipeline.run()
    run_dir = Path(summary["run_dir"])
    types = set()
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        evt = json.loads(line)
        types.add(evt["event_type"])
    assert "pipeline_started" in types
    assert "pipeline_completed" in types
