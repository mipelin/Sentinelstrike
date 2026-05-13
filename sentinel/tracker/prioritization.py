"""Track priority engine — deterministic, explainable scoring."""

from __future__ import annotations

from .track import MovementState, TrackState

# Speed thresholds (px/s) — configurable via TrackerConfig
_DEFAULT_STATIONARY_THRESHOLD = 3.0

# Class-based base priority (0-40 range)
_CLASS_BASE_SCORE: dict[str, float] = {
    "person": 30.0,
    "bicycle": 15.0,
    "motorcycle": 20.0,
    "car": 10.0,
    "truck": 12.0,
    "bus": 10.0,
    "van": 10.0,
}

_DEFAULT_CLASS_SCORE = 5.0

# Movement state multiplier (0-35 range)
_MOVEMENT_SCORE: dict[MovementState, float] = {
    MovementState.STATIONARY: 0.0,
    MovementState.SLOW_MOVING: 10.0,
    MovementState.MOVING: 20.0,
    MovementState.FAST_MOVING: 35.0,
}

# Reacquisition bonus (0-15 range)
_REACQUIRED_BONUS_PER_INSTANCE = 5.0
_MAX_REACQUIRED_BONUS = 15.0

# Track age bonus (0-10 range) — rewards tracks that persist
def _age_score(age_frames: int) -> float:
    if age_frames < 3:
        return 0.0
    if age_frames < 10:
        return 3.0
    if age_frames < 30:
        return 6.0
    return 10.0

# Confidence bonus (0-5 range)
def _confidence_score(confidence: float) -> float:
    return confidence * 5.0


def compute_priority(
    trk: TrackState,
    stationary_threshold_px_s: float = _DEFAULT_STATIONARY_THRESHOLD,
) -> tuple[float, str]:
    """Compute priority_score (0-100) and priority_level for a track.

    Returns (score, level) where level is one of LOW, MEDIUM, HIGH, CRITICAL.
    """
    class_lower = trk.class_name.lower()
    base = _CLASS_BASE_SCORE.get(class_lower, _DEFAULT_CLASS_SCORE)
    movement = _MOVEMENT_SCORE.get(trk.movement_state, 0.0)
    reacquired_bonus = min(
        trk.reacquired_count * _REACQUIRED_BONUS_PER_INSTANCE,
        _MAX_REACQUIRED_BONUS,
    )
    age = _age_score(trk.age_frames)
    conf = _confidence_score(trk.smoothed_confidence)

    raw = base + movement + reacquired_bonus + age + conf
    score = max(0.0, min(100.0, raw))

    if score >= 70.0:
        level = "CRITICAL"
    elif score >= 45.0:
        level = "HIGH"
    elif score >= 25.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


def classify_suppression(
    trk: TrackState,
    suppress_after_frames: int = 50,
    stationary_threshold_px_s: float = _DEFAULT_STATIONARY_THRESHOLD,
) -> bool:
    """Determine if a track should be suppressed (reduced TAK/geo publishing).

    A track is suppressed when it is STATIONARY and has been stationary
    for longer than suppress_after_frames.
    """
    if trk.movement_state != MovementState.STATIONARY:
        return False
    if trk.stationary_frames < suppress_after_frames:
        return False
    return True


def update_track_priority(
    trk: TrackState,
    suppress_after_frames: int = 50,
    stationary_threshold_px_s: float = _DEFAULT_STATIONARY_THRESHOLD,
) -> None:
    """Compute and assign priority + suppression state to a track."""
    score, level = compute_priority(trk, stationary_threshold_px_s)
    trk.priority_score = score
    trk.priority_level = level
    trk.suppressed = classify_suppression(trk, suppress_after_frames, stationary_threshold_px_s)
