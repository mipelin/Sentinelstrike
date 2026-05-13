"""Tests for GeolocalizationRunner."""

from sentinel.common.event_bus import EventBus
from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, GeoPoint, Track, VehicleState
from sentinel.config.schema import GeolocalizerConfig
from sentinel.geolocalizer.runner import GeolocalizationRunner


def _track(track_id="trk_000001", conf=0.9, x1=100, y1=50, x2=300, y2=250):
    return Track(
        track_id=track_id,
        class_name="person",
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        last_seen_utc=utc_now_iso(),
    )


def _vehicle():
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80.0),
        heading_deg=90.0,
        groundspeed_mps=10.0,
        mode="SIMULATED",
        armed=False,
        battery_pct=90.0,
    )


def test_runner_creates_geo_observations_jsonl(tmp_path):
    cfg = GeolocalizerConfig(camera={"pitch_deg": -45.0})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = GeolocalizationRunner(config=cfg, mission_id="m1", run_dir=run_dir)
    runner.process_tracks([_track()], _vehicle())
    runner.close()
    assert (run_dir / "geo_observations.jsonl").exists()


def test_runner_creates_metrics_json(tmp_path):
    cfg = GeolocalizerConfig(camera={"pitch_deg": -45.0})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = GeolocalizationRunner(config=cfg, mission_id="m1", run_dir=run_dir)
    runner.process_tracks([_track()], _vehicle())
    runner.close()
    assert (run_dir / "geolocalizer_metrics.json").exists()


def test_runner_publishes_geo_observation_created():
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)
    cfg = GeolocalizerConfig(camera={"pitch_deg": -45.0})
    runner = GeolocalizationRunner(config=cfg, mission_id="m1", event_bus=bus)
    runner.process_tracks([_track()], _vehicle())
    runner.close()
    types = [e.event_type for e in events]
    assert "geo_observation_created" in types


def test_runner_publishes_geolocalization_completed():
    bus = EventBus()
    events = []
    bus.subscribe("*", events.append)
    cfg = GeolocalizerConfig(camera={"pitch_deg": -45.0})
    runner = GeolocalizationRunner(config=cfg, mission_id="m1", event_bus=bus)
    runner.process_tracks([_track()], _vehicle())
    runner.close()
    types = [e.event_type for e in events]
    assert "geolocalization_completed" in types


def test_runner_filters_by_min_track_confidence(tmp_path):
    cfg = GeolocalizerConfig(min_track_confidence=0.5, camera={"pitch_deg": -45.0})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = GeolocalizationRunner(config=cfg, mission_id="m1", run_dir=run_dir)

    low_conf = _track(track_id="trk_low", conf=0.3)
    high_conf = _track(track_id="trk_high", conf=0.9)
    results = runner.process_tracks([low_conf, high_conf], _vehicle())
    runner.close()

    assert len(results) == 1
    assert results[0].track_id == "trk_high"
