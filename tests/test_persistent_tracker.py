"""Tests for PersistentTrackManager."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.tracker.persistent import (
    PersistentTrackManager,
    _bbox_iou,
    _center_dist,
    _cosine_similarity,
    compute_color_histogram,
)


def _make_frame(h: int = 200, w: int = 300, color: tuple = (128, 128, 128)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _det(
    cls: str = "person",
    conf: float = 0.85,
    bbox: tuple = (50, 50, 100, 150),
) -> dict:
    return {"class": cls, "confidence": conf, "bbox": list(bbox)}


def _shifted_det(det: dict, dx: int, dy: int) -> dict:
    x1, y1, x2, y2 = det["bbox"]
    return {"class": det["class"], "confidence": det["confidence"], "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy]}


class TestBboxIoU:
    def test_no_overlap(self):
        assert _bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_full_overlap(self):
        assert abs(_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 0.01

    def test_partial_overlap(self):
        iou = _bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.5


class TestCenterDist:
    def test_same_center(self):
        assert _center_dist((0, 0, 10, 10), (2, 2, 8, 8)) < 0.01

    def test_far_apart(self):
        d = _center_dist((0, 0, 10, 10), (100, 100, 110, 110))
        assert d > 100.0


class TestCosineSimilarity:
    def test_identical(self):
        v = np.array([1, 0, 0], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.01

    def test_orthogonal(self):
        a = np.array([1, 0], dtype=np.float32)
        b = np.array([0, 1], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 0.01

    def test_none_returns_zero(self):
        assert _cosine_similarity(None, np.zeros(3)) == 0.0


class TestColorHistogram:
    def test_valid_crop(self):
        frame = _make_frame()
        hist = compute_color_histogram(frame, (10, 10, 50, 50))
        assert hist.shape == (128,)
        # Uniform frame → all histogram bins may be nonzero but should be normalized
        assert np.linalg.norm(hist) - 1.0 < 0.1 or np.linalg.norm(hist) < 0.01

    def test_invalid_crop(self):
        frame = _make_frame()
        hist = compute_color_histogram(frame, (0, 0, 0, 0))
        assert hist.shape == (128,)
        assert np.linalg.norm(hist) < 0.01


class TestTrackCreationAndPromotion:
    def test_tentative_before_min_hits(self):
        mgr = PersistentTrackManager(min_hits=3)
        frame = _make_frame()
        d = _det()
        r = mgr.update([d], frame, 0)
        assert r[0]["status"] == "TENTATIVE"
        assert r[0]["track_id"].startswith("TGT-")

    def test_active_after_min_hits(self):
        mgr = PersistentTrackManager(min_hits=3)
        frame = _make_frame()
        d = _det()
        for i in range(3):
            mgr.update([d], frame, i)
        tracks = mgr.get_metrics()
        assert tracks["active"] == 1

    def test_active_on_fourth_hit(self):
        mgr = PersistentTrackManager(min_hits=3)
        frame = _make_frame()
        d = _det()
        for i in range(4):
            r = mgr.update([d], frame, i)
        active = [t for t in r if t["status"] == "ACTIVE"]
        assert len(active) == 1


class TestTrackIdStability:
    def test_same_id_after_loss_and_reacquire(self):
        mgr = PersistentTrackManager(min_hits=2, max_lost_frames=10, iou_thresh=0.2, center_dist_thresh=200)
        frame = _make_frame()
        d = _det()

        # Build up to ACTIVE
        r1 = mgr.update([d], frame, 0)
        r2 = mgr.update([d], frame, 1)
        tgt_id = [t["track_id"] for t in r2 if t["status"] == "ACTIVE"][0]

        # Lose for a few frames
        for i in range(2, 5):
            mgr.update([], frame, i)

        # Reappear at same position
        r3 = mgr.update([d], frame, 5)
        reacq = [t for t in r3 if t["track_id"] == tgt_id]
        assert len(reacq) == 1
        assert reacq[0]["status"] == "REACQUIRED"


class TestLostTrackPredictedBbox:
    def test_predicted_bbox_while_lost(self):
        mgr = PersistentTrackManager(min_hits=2, max_lost_frames=10)
        frame = _make_frame()

        # Create track moving right
        d1 = _det(bbox=(50, 50, 100, 150))
        mgr.update([d1], frame, 0)
        d2 = _det(bbox=(60, 50, 110, 150))
        mgr.update([d2], frame, 1)

        # Lose it
        mgr.update([], frame, 2)

        # Check LOST track has predicted bbox
        all_tracks = mgr._tracks
        lost = [t for t in all_tracks if t.state == "LOST"]
        assert len(lost) == 1
        assert lost[0].predicted_bbox is not None


class TestStaleAfterMaxLost:
    def test_stale_after_max_lost_frames(self):
        mgr = PersistentTrackManager(min_hits=1, max_lost_frames=5)
        frame = _make_frame()
        d = _det(bbox=(50, 50, 100, 150))

        mgr.update([d], frame, 0)

        # Lose for max_lost_frames + 1
        for i in range(1, 7):
            mgr.update([], frame, i)

        stale = [t for t in mgr._tracks if t.state == "STALE"]
        assert len(stale) == 1


class TestDuplicateSuppression:
    def test_overlapping_tentative_removed(self):
        mgr = PersistentTrackManager(min_hits=3, duplicate_iou=0.3)
        frame = _make_frame()

        # Create active track
        d = _det(bbox=(50, 50, 100, 150))
        for i in range(4):
            mgr.update([d], frame, i)

        # Add overlapping detection at same position → should be suppressed
        d_overlap = _det(bbox=(55, 55, 105, 155))
        mgr.update([d, d_overlap], frame, 5)

        metrics = mgr.get_metrics()
        # Should only have 1 confirmed track
        assert metrics["unique_confirmed"] == 1


class TestClassAwareMatching:
    def test_person_does_not_match_car(self):
        mgr = PersistentTrackManager(min_hits=2, iou_thresh=0.1)
        frame = _make_frame()

        person = _det(cls="person", bbox=(50, 50, 100, 150))
        car = _det(cls="car", bbox=(55, 55, 105, 155))

        mgr.update([person], frame, 0)
        r = mgr.update([car], frame, 1)

        # Should create two separate tracks
        ids = set(t["track_id"] for t in r if t["track_id"])
        assert len(ids) >= 2


class TestTargetIdFormat:
    def test_sequential_ids(self):
        mgr = PersistentTrackManager(min_hits=1)
        frame = _make_frame()

        d1 = _det(cls="person", bbox=(10, 10, 50, 100))
        d2 = _det(cls="car", bbox=(100, 10, 200, 100))
        r = mgr.update([d1, d2], frame, 0)

        ids = [t["track_id"] for t in r]
        assert "TGT-001" in ids
        assert "TGT-002" in ids


class TestReacquiredStateTransitions:
    def test_active_to_lost_to_reacquired_to_active(self):
        mgr = PersistentTrackManager(min_hits=2, max_lost_frames=20, center_dist_thresh=200)
        frame = _make_frame()
        d = _det()

        # ACTIVE
        mgr.update([d], frame, 0)
        r = mgr.update([d], frame, 1)
        states = [t["status"] for t in r]
        assert "ACTIVE" in states

        # LOST
        mgr.update([], frame, 2)
        r = mgr.update([], frame, 3)
        states = [t["status"] for t in r]
        assert "LOST" in states

        # REACQUIRED
        r = mgr.update([d], frame, 4)
        states = [t["status"] for t in r]
        assert "REACQUIRED" in states

        # Back to ACTIVE on next hit
        r = mgr.update([d], frame, 5)
        states = [t["status"] for t in r]
        assert "ACTIVE" in states


class TestMultipleSimultaneousMatches:
    def test_three_dets_match_three_tracks(self):
        mgr = PersistentTrackManager(min_hits=2, iou_thresh=0.1)
        frame = _make_frame()

        d1 = _det(cls="person", bbox=(10, 10, 50, 100))
        d2 = _det(cls="car", bbox=(100, 10, 200, 100))
        d3 = _det(cls="truck", bbox=(200, 10, 300, 100))

        mgr.update([d1, d2, d3], frame, 0)
        r = mgr.update([d1, d2, d3], frame, 1)

        ids = set(t["track_id"] for t in r if t["track_id"])
        assert len(ids) == 3


class TestMetrics:
    def test_metrics_after_several_updates(self):
        mgr = PersistentTrackManager(min_hits=2, max_lost_frames=5)
        frame = _make_frame()

        d1 = _det(cls="person", bbox=(10, 10, 50, 100))
        d2 = _det(cls="car", bbox=(100, 10, 200, 100))

        mgr.update([d1, d2], frame, 0)
        mgr.update([d1, d2], frame, 1)
        mgr.update([d1], frame, 2)  # car lost
        mgr.update([d1], frame, 3)

        m = mgr.get_metrics()
        assert m["active"] >= 1
        assert m["total_tracks_created"] == 2
