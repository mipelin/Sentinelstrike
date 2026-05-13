"""Tests for SimpleIoUTracker."""

from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, Detection
from sentinel.tracker.simple_tracker import SimpleIoUTracker


def _det(frame_id: int = 0, class_name: str = "person", x1=0, y1=0, x2=10, y2=10, conf=0.9):
    return Detection(
        frame_id=frame_id,
        timestamp_utc=utc_now_iso(),
        class_name=class_name,
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_first_detection_creates_track():
    tracker = SimpleIoUTracker()
    tracks = tracker.update([_det()], 0, utc_now_iso())
    assert len(tracks) == 1
    assert tracks[0].track_id == "trk_000001"


def test_similar_detection_maintains_track():
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracks = tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 1, ts)
    assert tracks[0].track_id == "trk_000001"


def test_distant_detection_creates_new_track():
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracks = tracker.update([_det(x1=500, y1=500, x2=510, y2=510)], 1, ts)
    ids = [t.track_id for t in tracks]
    assert "trk_000001" in ids
    assert "trk_000002" in ids


def test_track_without_detection_becomes_lost():
    tracker = SimpleIoUTracker(max_lost_frames=3)
    ts = utc_now_iso()
    tracker.update([_det()], 0, ts)
    tracks = tracker.update([], 1, ts)
    assert len(tracks) == 1
    assert tracks[0].status == "lost"


def test_track_exceeds_max_lost_becomes_terminated():
    tracker = SimpleIoUTracker(max_lost_frames=2)
    ts = utc_now_iso()
    tracker.update([_det()], 0, ts)
    tracker.update([], 1, ts)
    tracker.update([], 2, ts)
    tracker.update([], 3, ts)
    active = tracker.get_active_tracks()
    all_trk = tracker.get_all_tracks()
    assert len(active) == 0
    assert any(t.status == "terminated" for t in all_trk)


def test_min_confidence_filters():
    tracker = SimpleIoUTracker(min_confidence=0.5)
    tracks = tracker.update([_det(conf=0.3)], 0, utc_now_iso())
    assert len(tracks) == 0


def test_reset_clears_state():
    tracker = SimpleIoUTracker()
    tracker.update([_det()], 0, utc_now_iso())
    tracker.reset()
    tracks = tracker.update([_det()], 0, utc_now_iso())
    assert tracks[0].track_id == "trk_000001"


def test_lost_track_has_lost_frames_greater_than_zero():
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det()], 0, ts)
    tracks = tracker.update([], 1, ts)
    assert tracks[0].status == "lost"
    assert tracks[0].lost_frames == 1


def test_active_track_has_lost_frames_zero_after_reacquire():
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([], 1, ts)  # lost
    tracks = tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 2, ts)  # reacquire
    assert tracks[0].status == "active"
    assert tracks[0].lost_frames == 0


def test_first_seen_and_last_seen_are_set_correctly():
    tracker = SimpleIoUTracker()
    t0 = "2026-01-01T00:00:00Z"
    t1 = "2026-01-01T00:00:01Z"
    t2 = "2026-01-01T00:00:02Z"
    tracker.update([_det()], 0, t0)
    # Detection in frame 1 — last_seen should advance
    tracks = tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 1, t1)
    assert tracks[0].first_seen_utc == t0
    assert tracks[0].last_seen_utc == t1
    # No detection in frame 2 — last_seen must stay the same
    tracks = tracker.update([], 2, t2)
    assert tracks[0].first_seen_utc == t0
    assert tracks[0].last_seen_utc == t1


def test_terminated_track_has_lost_frames_above_threshold():
    tracker = SimpleIoUTracker(max_lost_frames=1)
    ts = utc_now_iso()
    tracker.update([_det()], 0, ts)
    tracker.update([], 1, ts)  # lost_frames=1 (<= max_lost_frames), status=lost
    tracker.update([], 2, ts)  # lost_frames=2 (> max_lost_frames), status=terminated
    all_trk = tracker.get_all_tracks()
    terminated = [t for t in all_trk if t.status == "terminated"]
    assert len(terminated) == 1
    assert terminated[0].lost_frames == 2
