"""Tests for track/observation/TAK message explosion prevention."""

from __future__ import annotations

from sentinel.common.types import BoundingBox, Detection, GeoObservation, GeoPoint, Track, VehicleState
from sentinel.config.schema import GeolocalizerConfig, TakConfig, TrackerConfig
from sentinel.geolocalizer.runner import GeolocalizationRunner
from sentinel.tak_bridge.bridge import TakBridge
from sentinel.tracker.runner import TrackingRunner
from sentinel.tracker.simple_tracker import SimpleIoUTracker


def _det(frame_id: int, class_name: str = "person", confidence: float = 0.9, **kwargs) -> Detection:
    return Detection(
        frame_id=frame_id,
        timestamp_utc="2026-01-01T00:00:00Z",
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=BoundingBox(x1=100, y1=100, x2=200, y2=200),
        **kwargs,
    )


def _det_at(frame_id: int, x1: int, y1: int, x2: int, y2: int, **kwargs) -> Detection:
    return Detection(
        frame_id=frame_id,
        timestamp_utc="2026-01-01T00:00:00Z",
        class_name="person",
        confidence=0.9,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        **kwargs,
    )


def _vehicle_state() -> VehicleState:
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc="2026-01-01T00:00:00Z",
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80.0),
        heading_deg=0.0,
        groundspeed_mps=10.0,
        battery_pct=80.0,
    )


# --- Tracker: frame-active tracks only ---

class TestTrackerFrameActive:
    def test_update_returns_all_non_terminated(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        det = _det(0)
        tracks = tracker.update([det], 0, "2026-01-01T00:00:00Z")
        assert len(tracks) == 1

        # No detections next frame — track becomes lost but still returned
        tracks = tracker.update([], 1, "2026-01-01T00:00:01Z")
        assert len(tracks) == 1
        assert tracks[0].status == "lost"

    def test_frame_active_excludes_lost(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        det = _det(0)
        tracker.update([det], 0, "2026-01-01T00:00:00Z")

        # Frame 1: no detections — track becomes lost, not frame-active
        tracker.update([], 1, "2026-01-01T00:00:01Z")
        frame_active = tracker.get_frame_active_tracks()
        assert len(frame_active) == 0

    def test_frame_active_includes_matched(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        det0 = _det(0)
        tracker.update([det0], 0, "2026-01-01T00:00:00Z")

        det1 = _det(1)
        tracker.update([det1], 1, "2026-01-01T00:00:01Z")
        frame_active = tracker.get_frame_active_tracks()
        assert len(frame_active) == 1
        assert frame_active[0].status == "active"

    def test_frame_active_includes_new_tracks(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        det = _det(0)
        tracker.update([det], 0, "2026-01-01T00:00:00Z")
        frame_active = tracker.get_frame_active_tracks()
        assert len(frame_active) == 1

    def test_persistent_detection_one_track(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        det = _det(0)

        for frame_id in range(10):
            tracker.update([det], frame_id, f"2026-01-01T00:00:0{frame_id}Z")

        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0].age_frames == 10

    def test_multiple_persistent_detections_few_tracks(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        dets = [
            _det_at(0, 100, 100, 200, 200),
            _det_at(0, 300, 300, 400, 400),
            _det_at(0, 500, 500, 600, 600),
        ]

        for frame_id in range(50):
            tracker.update(dets, frame_id, f"2026-01-01T00:00:{frame_id:02d}Z")

        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 3

        frame_active = tracker.get_frame_active_tracks()
        assert len(frame_active) == 3


class TestTrackingRunnerFrameActive:
    def test_runner_returns_frame_active_only(self, tmp_path):
        config = TrackerConfig(enabled=True, iou_threshold=0.2, save_tracks_jsonl=False)
        runner = TrackingRunner(config, mission_id="test", run_dir=tmp_path)

        det = _det(0)
        tracks_f0 = runner.process_frame_detections(0, "2026-01-01T00:00:00Z", [det])
        assert len(tracks_f0) == 1

        # Frame 1: no detections — runner should return empty (no frame-active)
        tracks_f1 = runner.process_frame_detections(1, "2026-01-01T00:00:01Z", [])
        assert len(tracks_f1) == 0

        runner.close(frame_count=2)

    def test_runner_writes_all_to_jsonl(self, tmp_path):
        config = TrackerConfig(enabled=True, iou_threshold=0.2, save_tracks_jsonl=True)
        runner = TrackingRunner(config, mission_id="test", run_dir=tmp_path)

        det = _det(0)
        runner.process_frame_detections(0, "2026-01-01T00:00:00Z", [det])
        runner.process_frame_detections(1, "2026-01-01T00:00:01Z", [])

        runner.close(frame_count=2)

        tracks_jsonl = tmp_path / "tracks.jsonl"
        assert tracks_jsonl.exists()
        lines = tracks_jsonl.read_text().strip().split("\n")
        # Frame 0: 1 track created. Frame 1: 1 track becomes lost. = 2 lines
        assert len(lines) == 2


# --- Geolocalizer throttling ---

class TestGeolocalizerThrottle:
    def test_first_observation_always_published(self, tmp_path):
        config = GeolocalizerConfig(
            enabled=True,
            publish_active_only=True,
            min_publish_interval_s=100.0,
            min_movement_m=1000.0,
            camera={"width_px": 640, "height_px": 480},
        )
        runner = GeolocalizationRunner(config, mission_id="test", run_dir=tmp_path)
        track = Track(
            track_id="trk_001",
            class_name="person",
            confidence=0.9,
            bbox_xyxy=BoundingBox(x1=320, y1=240, x2=330, y2=250),
            status="active",
            last_seen_utc="2026-01-01T00:00:00Z",
        )
        obs = runner.process_tracks([track], _vehicle_state(), "2026-01-01T00:00:00Z")
        assert len(obs) == 1
        runner.close()

    def test_throttle_by_interval(self, tmp_path):
        config = GeolocalizerConfig(
            enabled=True,
            publish_active_only=True,
            min_publish_interval_s=100.0,
            min_movement_m=0.0,
            camera={"width_px": 640, "height_px": 480},
        )
        runner = GeolocalizationRunner(config, mission_id="test", run_dir=tmp_path)
        track = Track(
            track_id="trk_001",
            class_name="person",
            confidence=0.9,
            bbox_xyxy=BoundingBox(x1=320, y1=240, x2=330, y2=250),
            status="active",
            last_seen_utc="2026-01-01T00:00:00Z",
        )
        vs = _vehicle_state()
        # First: published
        obs1 = runner.process_tracks([track], vs, "2026-01-01T00:00:00Z")
        assert len(obs1) == 1
        # Second immediately: suppressed
        obs2 = runner.process_tracks([track], vs, "2026-01-01T00:00:01Z")
        assert len(obs2) == 0
        assert runner.suppressed_count == 1
        runner.close()

    def test_throttle_by_max_per_track(self, tmp_path):
        config = GeolocalizerConfig(
            enabled=True,
            publish_active_only=True,
            min_publish_interval_s=0.0,
            min_movement_m=0.0,
            max_observations_per_track=3,
            camera={"width_px": 640, "height_px": 480},
        )
        runner = GeolocalizationRunner(config, mission_id="test", run_dir=tmp_path)
        track = Track(
            track_id="trk_001",
            class_name="person",
            confidence=0.9,
            bbox_xyxy=BoundingBox(x1=320, y1=240, x2=330, y2=250),
            status="active",
            last_seen_utc="2026-01-01T00:00:00Z",
        )
        vs = _vehicle_state()
        for _ in range(5):
            runner.process_tracks([track], vs, "2026-01-01T00:00:00Z")
        assert runner.suppressed_count == 2
        runner.close()

    def test_lost_track_suppressed_when_active_only(self, tmp_path):
        config = GeolocalizerConfig(
            enabled=True,
            publish_active_only=True,
            min_publish_interval_s=0.0,
            min_movement_m=0.0,
            camera={"width_px": 640, "height_px": 480},
        )
        runner = GeolocalizationRunner(config, mission_id="test", run_dir=tmp_path)
        lost_track = Track(
            track_id="trk_001",
            class_name="person",
            confidence=0.9,
            bbox_xyxy=BoundingBox(x1=320, y1=240, x2=330, y2=250),
            status="lost",
            last_seen_utc="2026-01-01T00:00:00Z",
        )
        obs = runner.process_tracks([lost_track], _vehicle_state(), "2026-01-01T00:00:00Z")
        assert len(obs) == 0
        runner.close()

    def test_throttle_disabled_publishes_all(self, tmp_path):
        config = GeolocalizerConfig(
            enabled=True,
            publish_active_only=False,
            min_publish_interval_s=0.0,
            min_movement_m=0.0,
            camera={"width_px": 640, "height_px": 480},
        )
        runner = GeolocalizationRunner(config, mission_id="test", run_dir=tmp_path)
        track = Track(
            track_id="trk_001",
            class_name="person",
            confidence=0.9,
            bbox_xyxy=BoundingBox(x1=320, y1=240, x2=330, y2=250),
            status="active",
            last_seen_utc="2026-01-01T00:00:00Z",
        )
        vs = _vehicle_state()
        for _ in range(10):
            runner.process_tracks([track], vs, "2026-01-01T00:00:00Z")
        assert runner.suppressed_count == 0
        runner.close()


# --- TAK throttling ---

class TestTakThrottle:
    def test_first_vehicle_always_sent(self, tmp_path):
        config = TakConfig(
            enabled=True,
            mode="dry_run",
            save_tak_messages_jsonl=False,
            publish_vehicle_every_s=5.0,
            publish_track_every_s=3.0,
        )
        bridge = TakBridge(config, mission_id="test", run_dir=tmp_path)
        vs = _vehicle_state()
        assert bridge.send_vehicle_state(vs) is True
        bridge.close()

    def test_vehicle_throttle_by_interval(self, tmp_path):
        config = TakConfig(
            enabled=True,
            mode="dry_run",
            save_tak_messages_jsonl=False,
            publish_vehicle_every_s=100.0,
            publish_track_every_s=0.0,
        )
        bridge = TakBridge(config, mission_id="test", run_dir=tmp_path)
        vs = _vehicle_state()
        assert bridge.send_vehicle_state(vs) is True
        assert bridge.send_vehicle_state(vs) is False
        assert bridge.suppressed_count == 1
        bridge.close()

    def test_observation_throttle_by_track_interval(self, tmp_path):
        config = TakConfig(
            enabled=True,
            mode="dry_run",
            save_tak_messages_jsonl=False,
            publish_vehicle_every_s=0.0,
            publish_track_every_s=100.0,
        )
        bridge = TakBridge(config, mission_id="test", run_dir=tmp_path)
        obs = GeoObservation(
            observation_id="obs_001",
            mission_id="test",
            track_id="trk_001",
            class_name="person",
            timestamp_utc="2026-01-01T00:00:00Z",
            estimated_location=GeoPoint(lat=38.0, lon=-8.0),
            confidence=0.9,
        )
        assert bridge.send_geo_observation(obs) is True
        assert bridge.send_geo_observation(obs) is False
        assert bridge.suppressed_count == 1
        bridge.close()

    def test_max_messages_per_run(self, tmp_path):
        config = TakConfig(
            enabled=True,
            mode="dry_run",
            save_tak_messages_jsonl=False,
            publish_vehicle_every_s=0.0,
            publish_track_every_s=0.0,
            max_messages_per_run=2,
        )
        bridge = TakBridge(config, mission_id="test", run_dir=tmp_path)
        obs = GeoObservation(
            observation_id="obs_001",
            mission_id="test",
            track_id="trk_001",
            class_name="person",
            timestamp_utc="2026-01-01T00:00:00Z",
            estimated_location=GeoPoint(lat=38.0, lon=-8.0),
            confidence=0.9,
        )
        assert bridge.send_geo_observation(obs) is True
        assert bridge.send_geo_observation(obs) is True
        assert bridge.send_geo_observation(obs) is False
        assert bridge.suppressed_count == 1
        bridge.close()

    def test_different_tracks_not_throttled_together(self, tmp_path):
        config = TakConfig(
            enabled=True,
            mode="dry_run",
            save_tak_messages_jsonl=False,
            publish_vehicle_every_s=0.0,
            publish_track_every_s=100.0,
        )
        bridge = TakBridge(config, mission_id="test", run_dir=tmp_path)
        obs_a = GeoObservation(
            observation_id="obs_a",
            mission_id="test",
            track_id="trk_a",
            class_name="person",
            timestamp_utc="2026-01-01T00:00:00Z",
            estimated_location=GeoPoint(lat=38.0, lon=-8.0),
            confidence=0.9,
        )
        obs_b = GeoObservation(
            observation_id="obs_b",
            mission_id="test",
            track_id="trk_b",
            class_name="car",
            timestamp_utc="2026-01-01T00:00:00Z",
            estimated_location=GeoPoint(lat=38.01, lon=-8.01),
            confidence=0.8,
        )
        assert bridge.send_geo_observation(obs_a) is True
        assert bridge.send_geo_observation(obs_b) is True
        bridge.close()


# --- Integration: no explosion ---

class TestNoExplosion:
    def test_300_frames_reasonable_output(self, tmp_path):
        """Simulate 300 frames with persistent detections and verify output is bounded."""
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        total_frame_active = 0
        all_track_ids: set[str] = set()

        dets = [
            _det_at(0, 100, 100, 200, 200),
            _det_at(0, 300, 300, 400, 400),
            _det_at(0, 500, 500, 600, 600),
        ]

        for frame_id in range(300):
            tracker.update(dets, frame_id, f"2026-01-01T00:{frame_id // 60:02d}:{frame_id % 60:02d}Z")
            frame_active = tracker.get_frame_active_tracks()
            total_frame_active += len(frame_active)
            for trk in frame_active:
                all_track_ids.add(trk.track_id)

        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 3, f"Expected 3 unique tracks, got {len(all_tracks)}"
        assert len(all_track_ids) == 3
        # Frame-active per frame should be exactly 3 (all matched)
        assert total_frame_active == 300 * 3
