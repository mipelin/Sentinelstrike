"""Tests for track confirmation, duplicate suppression, and bbox smoothing."""

from sentinel.common.types import BoundingBox, Detection
from sentinel.common.time import utc_now_iso
from sentinel.tracker.simple_tracker import SimpleIoUTracker
from sentinel.tracker.track import ConfirmationState


def _det(x1=100, y1=100, x2=200, y2=200, class_name="person", confidence=0.8, frame_id=0):
    return Detection(
        frame_id=frame_id,
        timestamp_utc=utc_now_iso(),
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_same_detection_10_frames_one_confirmed_track():
    """Same detection repeated 10 frames → 1 confirmed track."""
    tracker = SimpleIoUTracker(min_hits_to_confirm=3)
    for i in range(10):
        tracker.update([_det(frame_id=i)], i, utc_now_iso())
    tracks = tracker.get_all_tracks()
    assert len(tracks) == 1
    assert tracks[0].metadata["confirmed"] is True
    assert tracks[0].status == "active"


def test_single_frame_detection_tentative():
    """1-frame detection → tentative track, not returned when filter on."""
    tracker = SimpleIoUTracker(
        min_hits_to_confirm=3,
        publish_tentative_tracks=False,
    )
    result = tracker.update([_det(confidence=0.5)], 0, utc_now_iso())
    # Not returned in the filtered output
    assert len(result) == 0
    # Still exists internally
    all_tracks = tracker.get_all_tracks()
    assert len(all_tracks) == 1
    assert all_tracks[0].metadata["confirmed"] is False


def test_class_aware_no_cross_match():
    """person detection should not match car track."""
    tracker = SimpleIoUTracker(
        min_hits_to_confirm=1,
        class_aware_matching=True,
    )
    # Create a person track
    tracker.update([_det(class_name="person", frame_id=0)], 0, utc_now_iso())
    # Send car detection at same location
    tracker.update([_det(class_name="car", frame_id=1)], 1, utc_now_iso())
    tracks = tracker.get_all_tracks()
    classes = {t.class_name for t in tracks}
    assert "person" in classes
    assert "car" in classes
    assert len(tracks) == 2


def test_lost_track_reacquired_without_new_id():
    """Lost track reappears nearby → reacquired, same track ID."""
    tracker = SimpleIoUTracker(
        max_lost_frames=10,
        min_hits_to_confirm=1,
        publish_tentative_tracks=True,
        reacquire_window_frames=20,
        duplicate_distance_px=80,
        duplicate_iou_threshold=0.1,
        iou_threshold=0.3,
    )
    # Frame 0: create track
    tracker.update([_det(x1=100, y1=100, x2=200, y2=200, frame_id=0)], 0, utc_now_iso())
    first_tracks = tracker.get_active_tracks()
    assert len(first_tracks) == 1
    track_id = first_tracks[0].track_id

    # Frames 1-3: no detection → track becomes lost
    for i in range(1, 4):
        tracker.update([], i, utc_now_iso())

    # Frame 4: detection reappears nearby
    tracker.update([_det(x1=105, y1=100, x2=205, y2=200, frame_id=4)], 4, utc_now_iso())
    active = tracker.get_active_tracks()
    assert len(active) == 1
    assert active[0].track_id == track_id
    assert active[0].metadata["reacquired_count"] >= 1


def test_bbox_smoothing_reduces_jitter():
    """Bbox smoothing should produce a smoothed display bbox."""
    tracker = SimpleIoUTracker(
        min_hits_to_confirm=1,
        publish_tentative_tracks=True,
        bbox_smoothing_enabled=True,
        bbox_ema_alpha=0.35,
    )
    # Frame 0
    tracker.update([_det(x1=100, y1=100, x2=200, y2=200, frame_id=0)], 0, utc_now_iso())
    # Frame 1: slightly shifted
    tracker.update([_det(x1=108, y1=103, x2=208, y2=203, frame_id=1)], 1, utc_now_iso())
    tracks = tracker.get_active_tracks()
    assert len(tracks) == 1
    meta = tracks[0].metadata
    # The track should have smoothing metadata
    assert "smoothed_confidence" in meta
    assert meta["track_quality_score"] > 0


def test_high_confidence_fast_confirm():
    """High confidence detection should be immediately confirmed."""
    tracker = SimpleIoUTracker(
        min_hits_to_confirm=5,
        fast_confirm_confidence=0.85,
        publish_tentative_tracks=True,
    )
    tracker.update([_det(confidence=0.95)], 0, utc_now_iso())
    tracks = tracker.get_all_tracks()
    assert len(tracks) == 1
    assert tracks[0].metadata["confirmed"] is True


def test_low_quality_tracks_still_tracked():
    """Low-quality tentative tracks exist but aren't returned when filtered."""
    tracker = SimpleIoUTracker(
        min_hits_to_confirm=10,
        publish_tentative_tracks=False,
    )
    for i in range(3):
        tracker.update([_det(confidence=0.3, frame_id=i)], i, utc_now_iso())
    # Not returned from update (filtered)
    result = tracker.update([_det(confidence=0.3, frame_id=3)], 3, utc_now_iso())
    confirmed = [t for t in result if t.metadata.get("confirmed")]
    assert len(confirmed) == 0
    # Still exists internally
    all_tracks = tracker.get_all_tracks()
    tentative = [t for t in all_tracks if not t.metadata.get("confirmed")]
    assert len(tentative) == 1
