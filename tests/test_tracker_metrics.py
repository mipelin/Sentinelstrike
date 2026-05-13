"""Tests for tracker metrics."""

from sentinel.common.types import Track
from sentinel.tracker.metrics import compute_tracker_metrics


def _track(class_name: str = "person", status: str = "active", age_frames: int = 5) -> Track:
    return Track(
        track_id="trk_000001",
        class_name=class_name,
        confidence=0.8,
        status=status,
        age_frames=age_frames,
        last_seen_utc="2025-01-01T00:00:00Z",
    )


def test_metrics_counts():
    tracks = [
        _track(status="active"),
        _track(status="lost", age_frames=10),
        _track(status="terminated", age_frames=20),
    ]
    m = compute_tracker_metrics(tracks, frame_count=30)
    assert m["total_tracks"] == 3
    assert m["active_tracks"] == 1
    assert m["lost_tracks"] == 1
    assert m["terminated_tracks"] == 1


def test_class_distribution():
    tracks = [_track(class_name="car"), _track(class_name="person"), _track(class_name="car")]
    m = compute_tracker_metrics(tracks, frame_count=10)
    assert m["classes"]["car"] == 2
    assert m["classes"]["person"] == 1


def test_average_age():
    tracks = [_track(age_frames=10), _track(age_frames=20)]
    m = compute_tracker_metrics(tracks, frame_count=10)
    assert m["average_track_age_frames"] == 15.0


def test_no_tracks_average_zero():
    m = compute_tracker_metrics([], frame_count=10)
    assert m["average_track_age_frames"] == 0
    assert m["total_tracks"] == 0
