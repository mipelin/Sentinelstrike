"""Tests for tracker motion prediction, reacquisition, and metadata."""

from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, Detection
from sentinel.tracker.simple_tracker import SimpleIoUTracker


def _det(frame_id=0, x1=0, y1=0, x2=10, y2=10, cls="person", conf=0.9):
    return Detection(
        frame_id=frame_id,
        timestamp_utc=utc_now_iso(),
        class_name=cls,
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# --- Velocity tracking ---

def test_velocity_updated_on_match():
    tracker = SimpleIoUTracker(iou_threshold=0.2)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracks = tracker.update([_det(x1=5, y1=3, x2=15, y2=13)], 1, ts)
    # Center moved from (5,5) to (10,8) → velocity (5,3)
    assert tracks[0].metadata["velocity"] == [5.0, 3.0]


def test_velocity_zero_first_frame():
    tracker = SimpleIoUTracker()
    tracks = tracker.update([_det()], 0, utc_now_iso())
    assert tracks[0].metadata["velocity"] == [0.0, 0.0]


def test_consecutive_hits_increments():
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 1, ts)
    tracks = tracker.update([_det(x1=2, y1=2, x2=12, y2=12)], 2, ts)
    assert tracks[0].metadata["consecutive_hits"] == 3


def test_consecutive_misses_on_lost():
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det()], 0, ts)
    tracks = tracker.update([], 1, ts)
    assert tracks[0].metadata["consecutive_hits"] == 0
    assert tracks[0].metadata["consecutive_misses"] == 1


# --- Prediction ---

def test_prediction_generates_predicted_bbox():
    tracker = SimpleIoUTracker(prediction_enabled=True, max_lost_frames=10)
    ts = utc_now_iso()
    # Overlapping detections to build velocity via IoU match
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)  # +5px right
    # Lost frame — should predict (velocity=(5,0))
    tracks = tracker.update([], 2, ts)
    assert tracks[0].metadata["predicted_bbox"] is not None
    assert tracks[0].metadata["prediction_age"] == 1


def test_prediction_disabled_no_predicted_bbox():
    tracker = SimpleIoUTracker(prediction_enabled=False, max_lost_frames=10)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)
    tracks = tracker.update([], 2, ts)
    assert tracks[0].metadata["predicted_bbox"] is None


def test_prediction_age_increments():
    tracker = SimpleIoUTracker(prediction_enabled=True, max_lost_frames=10)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)  # build velocity
    tracker.update([], 2, ts)
    tracks = tracker.update([], 3, ts)
    assert tracks[0].metadata["prediction_age"] == 2


def test_prediction_clears_on_reacquisition():
    tracker = SimpleIoUTracker(prediction_enabled=True, max_lost_frames=10)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)  # build velocity
    tracker.update([], 2, ts)  # lost, predicted
    tracks = tracker.update([_det(x1=10, y1=0, x2=20, y2=10)], 3, ts)  # reacquired (overlaps predicted)
    assert tracks[0].metadata["predicted_bbox"] is None
    assert tracks[0].metadata["prediction_age"] == 0


# --- Reacquisition ---

def test_reacquisition_preserves_id():
    tracker = SimpleIoUTracker(max_lost_frames=10)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([], 1, ts)  # lost
    tracks = tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 2, ts)
    assert tracks[0].track_id == "trk_000001"
    assert tracks[0].status == "active"


def test_reacquisition_increments_count():
    tracker = SimpleIoUTracker(max_lost_frames=10)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([], 1, ts)  # lost
    tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 2, ts)  # reacquired
    all_tracks = tracker.get_all_tracks()
    active = [t for t in all_tracks if t.status == "active"]
    assert active[0].metadata["reacquired_count"] == 1


def test_reacquisition_prediction_error():
    tracker = SimpleIoUTracker(
        prediction_enabled=True,
        max_lost_frames=10,
        iou_threshold=0.1,
        distance_threshold_px=100.0,
        iou_weight=0.5,
        distance_weight=0.5,
    )
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)  # build velocity (5,0)
    tracker.update([], 2, ts)  # lost, predicted at +5 from last
    tracks = tracker.update([_det(x1=12, y1=0, x2=22, y2=10)], 3, ts)  # reacquired near prediction
    active = [t for t in tracks if t.status == "active"]
    assert len(active) == 1
    err = active[0].metadata.get("last_prediction_error_px")
    assert err is not None
    assert err > 0  # not exactly at predicted position


# --- Hybrid association with distance ---

def test_distance_weight_helps_nearby_detection():
    tracker = SimpleIoUTracker(
        iou_threshold=0.1,
        distance_threshold_px=100.0,
        iou_weight=0.5,
        distance_weight=0.5,
    )
    ts = utc_now_iso()
    # Create track at (0,0)-(10,10)
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    # Detection nearby but no IoU overlap: (12,12)-(22,22)
    # Pure IoU would miss this, but distance component should help
    tracks = tracker.update([_det(x1=12, y1=12, x2=22, y2=22)], 1, ts)
    # With hybrid scoring, the nearby detection should match the existing track
    active = [t for t in tracks if t.status == "active"]
    assert len(active) == 1
    assert active[0].track_id == "trk_000001"


def test_pure_iou_misses_nearby_detection():
    tracker = SimpleIoUTracker(iou_threshold=0.1, iou_weight=1.0, distance_weight=0.0)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracks = tracker.update([_det(x1=12, y1=12, x2=22, y2=22)], 1, ts)
    # With pure IoU (default), nearby but non-overlapping → new track
    assert len(tracks) == 2


# --- Prediction-enhanced association ---

def test_prediction_enhanced_association():
    tracker = SimpleIoUTracker(
        prediction_enabled=True,
        iou_threshold=0.1,
        distance_threshold_px=100.0,
        iou_weight=0.5,
        distance_weight=0.5,
        max_lost_frames=10,
    )
    ts = utc_now_iso()
    # Build velocity via overlapping detections (5px/frame rightward)
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)
    tracker.update([], 2, ts)  # lost → predicts at +5 from (15,0)
    tracker.update([], 3, ts)  # still lost → predicts at +10
    # Detection near predicted position (should match via distance)
    tracks = tracker.update([_det(x1=20, y1=0, x2=30, y2=10)], 4, ts)
    active = [t for t in tracks if t.status == "active"]
    assert len(active) == 1
    assert active[0].track_id == "trk_000001"


# --- Metadata presence ---

def test_metadata_has_all_fields():
    tracker = SimpleIoUTracker(prediction_enabled=True, max_lost_frames=10)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=5, x2=15, y2=15)], 1, ts)
    tracker.update([], 2, ts)
    tracks = tracker.update([], 3, ts)
    meta = tracks[0].metadata
    assert "velocity" in meta
    assert "predicted_bbox" in meta
    assert "consecutive_hits" in meta
    assert "consecutive_misses" in meta
    assert "reacquired_count" in meta
    assert "prediction_age" in meta


def test_backward_compat_default_tracker():
    """Default SimpleIoUTracker produces same behavior as before."""
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracks = tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 1, ts)
    assert tracks[0].track_id == "trk_000001"
    assert tracks[0].lost_frames == 0
    assert tracks[0].status == "active"
