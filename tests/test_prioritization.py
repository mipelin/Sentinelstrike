"""Tests for motion classification, prioritization, and static suppression."""

from __future__ import annotations

from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, Detection, Track
from sentinel.tracker.metrics import compute_tracker_metrics
from sentinel.tracker.prioritization import (
    MovementState,
    classify_suppression,
    compute_priority,
    update_track_priority,
)
from sentinel.tracker.simple_tracker import SimpleIoUTracker
from sentinel.tracker.track import TrackState


def _det(frame_id=0, x1=0, y1=0, x2=10, y2=10, cls="person", conf=0.9):
    return Detection(
        frame_id=frame_id,
        timestamp_utc=utc_now_iso(),
        class_name=cls,
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _make_track_state(
    class_name="car",
    confidence=0.8,
    velocity=(0.0, 0.0),
    movement_state=MovementState.STATIONARY,
    stationary_frames=0,
    reacquired_count=0,
    age_frames=1,
) -> TrackState:
    ts = TrackState(
        track_id="trk_000001",
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=BoundingBox(x1=100, y1=100, x2=120, y2=120),
        first_seen_utc=utc_now_iso(),
    )
    ts.velocity = velocity
    ts.movement_state = movement_state
    ts.stationary_frames = stationary_frames
    ts.reacquired_count = reacquired_count
    ts.age_frames = age_frames
    return ts


# --- Motion Classification ---

def test_stationary_object_detected():
    """An object that doesn't move should be classified as STATIONARY."""
    tracker = SimpleIoUTracker(iou_threshold=0.2, fps=5.0, stationary_speed_threshold_px_s=3.0)
    ts = utc_now_iso()
    # Same position every frame → stationary
    tracker.update([_det(x1=100, y1=100, x2=120, y2=120)], 0, ts)
    tracker.update([_det(x1=100, y1=100, x2=120, y2=120)], 1, ts)
    tracks = tracker.update([_det(x1=100, y1=100, x2=120, y2=120)], 2, ts)
    assert tracks[0].metadata["movement_state"] == "STATIONARY"


def test_moving_object_classified():
    """An object moving consistently should be classified as MOVING or FAST_MOVING."""
    tracker = SimpleIoUTracker(
        iou_threshold=0.1,
        fps=5.0,
        stationary_speed_threshold_px_s=3.0,
        distance_threshold_px=100.0,
        iou_weight=0.5,
        distance_weight=0.5,
    )
    ts = utc_now_iso()
    # Overlapping detections moving 5px right per frame → velocity builds up
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)
    tracker.update([_det(x1=10, y1=0, x2=20, y2=10)], 2, ts)
    tracker.update([_det(x1=15, y1=0, x2=25, y2=10)], 3, ts)
    tracker.update([_det(x1=20, y1=0, x2=30, y2=10)], 4, ts)
    tracks = tracker.update([_det(x1=25, y1=0, x2=35, y2=10)], 5, ts)
    state = tracks[0].metadata["movement_state"]
    assert state in ("MOVING", "FAST_MOVING")


def test_slow_moving_object():
    """An object barely moving should be SLOW_MOVING."""
    tracker = SimpleIoUTracker(iou_threshold=0.2, fps=5.0, stationary_speed_threshold_px_s=3.0)
    ts = utc_now_iso()
    # 1px per frame at 5fps → 5 px/s (just above threshold of 3)
    tracker.update([_det(x1=100, y1=100, x2=120, y2=120)], 0, ts)
    tracker.update([_det(x1=101, y1=100, x2=121, y2=120)], 1, ts)
    tracks = tracker.update([_det(x1=102, y1=100, x2=122, y2=120)], 2, ts)
    assert tracks[0].metadata["movement_state"] == "SLOW_MOVING"


def test_parked_car_becomes_stationary():
    """A car that doesn't move should accumulate stationary frames."""
    tracker = SimpleIoUTracker(iou_threshold=0.2, fps=5.0, stationary_speed_threshold_px_s=3.0)
    ts = utc_now_iso()
    for i in range(20):
        tracker.update([_det(x1=50, y1=50, x2=80, y2=80, cls="car")], i, ts)
    tracks = tracker.get_all_tracks()
    active = [t for t in tracks if t.status == "active"]
    assert len(active) == 1
    assert active[0].metadata["movement_state"] == "STATIONARY"
    assert active[0].metadata["stationary_frames"] > 0


def test_speed_in_metadata():
    """Speed values should be present in metadata."""
    tracker = SimpleIoUTracker(iou_threshold=0.2, fps=5.0, stationary_speed_threshold_px_s=3.0)
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracker.update([_det(x1=5, y1=0, x2=15, y2=10)], 1, ts)
    tracks = tracker.update([_det(x1=10, y1=0, x2=20, y2=10)], 2, ts)
    assert "average_speed_px_s" in tracks[0].metadata
    assert "displacement_px" in tracks[0].metadata
    assert tracks[0].metadata["average_speed_px_s"] > 0.0


# --- Priority Scoring ---

def test_parked_car_low_priority():
    """A stationary car should get LOW priority."""
    ts = _make_track_state(class_name="car", movement_state=MovementState.STATIONARY)
    score, level = compute_priority(ts)
    assert level == "LOW"
    assert score < 30


def test_moving_car_medium_or_high():
    """A moving car should score higher than a parked car."""
    parked = _make_track_state(class_name="car", movement_state=MovementState.STATIONARY)
    moving = _make_track_state(class_name="car", movement_state=MovementState.MOVING)
    score_parked, _ = compute_priority(parked)
    score_moving, level_moving = compute_priority(moving)
    assert score_moving > score_parked
    assert level_moving in ("MEDIUM", "HIGH")


def test_walking_person_high():
    """A moving person should get HIGH priority."""
    ts = _make_track_state(class_name="person", movement_state=MovementState.MOVING, velocity=(10, 5))
    score, level = compute_priority(ts)
    assert level in ("HIGH", "CRITICAL")
    assert score >= 45


def test_fast_vehicle_high():
    """A fast-moving vehicle should get HIGH or CRITICAL priority."""
    ts = _make_track_state(class_name="car", movement_state=MovementState.FAST_MOVING, velocity=(50, 30))
    score, level = compute_priority(ts)
    assert level in ("HIGH", "CRITICAL")


def test_reacquired_target_critical():
    """A reacquired moving target should get CRITICAL priority."""
    ts = _make_track_state(
        class_name="person",
        movement_state=MovementState.MOVING,
        reacquired_count=3,
        age_frames=20,
    )
    score, level = compute_priority(ts)
    assert level == "CRITICAL"
    assert score >= 75


def test_priority_increases_with_motion():
    """Priority score should increase as motion increases."""
    states = [
        MovementState.STATIONARY,
        MovementState.SLOW_MOVING,
        MovementState.MOVING,
        MovementState.FAST_MOVING,
    ]
    scores = []
    for ms in states:
        ts = _make_track_state(movement_state=ms)
        score, _ = compute_priority(ts)
        scores.append(score)
    # Each state should score >= previous
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1], f"{states[i]} scored {scores[i]} < {scores[i-1]}"


def test_confidence_affects_score():
    """Higher confidence should give a slightly higher score."""
    low_conf = _make_track_state(confidence=0.3, movement_state=MovementState.MOVING)
    high_conf = _make_track_state(confidence=0.95, movement_state=MovementState.MOVING)
    score_low, _ = compute_priority(low_conf)
    score_high, _ = compute_priority(high_conf)
    assert score_high > score_low


# --- Static Suppression ---

def test_stationary_suppression_works():
    """A track stationary for many frames should be suppressed."""
    ts = _make_track_state(
        movement_state=MovementState.STATIONARY,
        stationary_frames=60,
    )
    assert classify_suppression(ts, suppress_after_frames=50) is True


def test_moving_not_suppressed():
    """A moving track should never be suppressed."""
    ts = _make_track_state(movement_state=MovementState.MOVING, stationary_frames=100)
    assert classify_suppression(ts, suppress_after_frames=50) is False


def test_not_yet_stationary_enough():
    """A track with few stationary frames should not be suppressed."""
    ts = _make_track_state(
        movement_state=MovementState.STATIONARY,
        stationary_frames=10,
    )
    assert classify_suppression(ts, suppress_after_frames=50) is False


def test_update_track_priority_sets_fields():
    """update_track_priority should set priority_score, priority_level, and suppressed."""
    ts = _make_track_state(movement_state=MovementState.STATIONARY, stationary_frames=60)
    update_track_priority(ts, suppress_after_frames=50)
    assert ts.priority_score > 0
    assert ts.priority_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert ts.suppressed is True


# --- Metrics ---

def test_metrics_include_motion_fields():
    """compute_tracker_metrics should include motion and priority fields."""
    tracks = [
        Track(
            track_id="trk_001",
            class_name="person",
            confidence=0.9,
            status="active",
            age_frames=20,
            last_seen_utc=utc_now_iso(),
            metadata={
                "movement_state": "MOVING",
                "average_speed_px_s": 15.0,
                "priority_level": "HIGH",
                "suppressed": False,
            },
        ),
        Track(
            track_id="trk_002",
            class_name="car",
            confidence=0.7,
            status="active",
            age_frames=50,
            last_seen_utc=utc_now_iso(),
            metadata={
                "movement_state": "STATIONARY",
                "average_speed_px_s": 0.5,
                "priority_level": "LOW",
                "suppressed": True,
            },
        ),
    ]
    m = compute_tracker_metrics(tracks, frame_count=100)
    assert m["moving_tracks"] == 1
    assert m["stationary_tracks"] == 1
    assert m["suppressed_stationary_tracks"] == 1
    assert m["high_priority_tracks"] == 1
    assert m["average_track_speed_px_s"] > 0


# --- Integration: tracker produces priority metadata ---

def test_tracker_produces_priority_metadata():
    """End-to-end: tracker + runner should produce priority fields in metadata."""
    from sentinel.config.schema import TrackerConfig
    from sentinel.tracker.runner import TrackingRunner

    config = TrackerConfig(
        enabled=True,
        fps=5.0,
        stationary_speed_threshold_px_s=3.0,
        suppress_stationary_after_s=2.0,
    )
    runner = TrackingRunner(config=config, mission_id="test_pri")

    ts = utc_now_iso()
    # Stationary car for 15 frames (2s at 5fps = 10 frames threshold)
    for i in range(15):
        runner.process_frame_detections(
            frame_id=i,
            timestamp_utc=ts,
            detections=[_det(x1=100, y1=100, x2=120, y2=120, cls="car", conf=0.8)],
        )

    all_tracks = runner.get_all_tracks()
    assert len(all_tracks) == 1
    meta = all_tracks[0].metadata
    assert "movement_state" in meta
    assert "priority_score" in meta
    assert "priority_level" in meta
    assert "suppressed" in meta
    assert "average_speed_px_s" in meta
    assert meta["movement_state"] == "STATIONARY"
    assert meta["suppressed"] is True


# --- Dashboard ordering ---

def test_dashboard_cards_ordered_by_priority():
    """Target cards should be sorted by priority (CRITICAL > HIGH > MEDIUM > LOW)."""
    from sentinel.dashboard.state import build_target_cards

    artifacts = {
        "tracks": [
            {
                "track_id": "trk_low",
                "class_name": "car",
                "confidence": 0.5,
                "status": "active",
                "age_frames": 10,
                "lost_frames": 0,
                "last_seen_utc": "2025-01-01T00:00:00Z",
                "metadata": {
                    "reacquired_count": 0,
                    "frame_id": 5,
                    "history_px": [],
                    "priority_level": "LOW",
                    "priority_score": 15.0,
                    "suppressed": True,
                    "movement_state": "STATIONARY",
                    "average_speed_px_s": 0.0,
                    "displacement_px": 0.0,
                    "stationary_frames": 20,
                    "moving_frames": 0,
                },
            },
            {
                "track_id": "trk_critical",
                "class_name": "person",
                "confidence": 0.95,
                "status": "active",
                "age_frames": 50,
                "lost_frames": 0,
                "last_seen_utc": "2025-01-01T00:00:00Z",
                "metadata": {
                    "reacquired_count": 3,
                    "frame_id": 5,
                    "history_px": [],
                    "priority_level": "CRITICAL",
                    "priority_score": 85.0,
                    "suppressed": False,
                    "movement_state": "FAST_MOVING",
                    "average_speed_px_s": 45.0,
                    "displacement_px": 200.0,
                    "stationary_frames": 0,
                    "moving_frames": 40,
                },
            },
        ],
        "operator_decisions": [],
        "geo_observations": [],
    }

    cards = build_target_cards(artifacts)
    assert len(cards) == 2
    # CRITICAL should come first
    assert cards[0].priority_level == "CRITICAL"
    assert cards[0].track_id == "trk_critical"
    assert cards[1].priority_level == "LOW"
    assert cards[1].track_id == "trk_low"


def test_existing_tests_still_pass():
    """Verify that default SimpleIoUTracker still works as before."""
    tracker = SimpleIoUTracker()
    ts = utc_now_iso()
    tracker.update([_det(x1=0, y1=0, x2=10, y2=10)], 0, ts)
    tracks = tracker.update([_det(x1=1, y1=1, x2=11, y2=11)], 1, ts)
    assert tracks[0].track_id == "trk_000001"
    assert tracks[0].lost_frames == 0
    assert tracks[0].status == "active"
    # New metadata fields present
    assert "movement_state" in tracks[0].metadata
    assert "priority_score" in tracks[0].metadata
