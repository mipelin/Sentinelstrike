"""Persistent target identity tracker with appearance-based reacquisition.

Maintains stable TGT-NNN identities through temporary detection losses
and occlusions. Uses IoU, center distance, and optional color histogram
appearance matching for robust track association.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_dist(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ca = _bbox_center(a)
    cb = _bbox_center(b)
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def _predict_bbox(
    bbox: tuple[int, int, int, int],
    velocity: tuple[float, float],
    frames: int,
) -> tuple[int, int, int, int]:
    cx, cy = _bbox_center(bbox)
    hw = (bbox[2] - bbox[0]) / 2.0
    hh = (bbox[3] - bbox[1]) / 2.0
    ncx = cx + velocity[0] * frames
    ncy = cy + velocity[1] * frames
    return (
        max(int(round(ncx - hw)), 0),
        max(int(round(ncy - hh)), 0),
        max(int(round(ncx + hw)), 1),
        max(int(round(ncy + hh)), 1),
    )


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_color_histogram(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Compute L2-normalized H+S 2D histogram from bbox crop (128-dim)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(128, dtype=np.float32)

    try:
        import cv2

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 1e-8:
            hist /= norm
        return hist
    except ImportError:
        return np.zeros(128, dtype=np.float32)


def _ema_smooth(old: float, new: float, alpha: float) -> float:
    return alpha * new + (1 - alpha) * old


def _ema_bbox_smooth(
    old: tuple[int, int, int, int],
    new: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    return (
        max(int(round(_ema_smooth(old[0], new[0], alpha))), 0),
        max(int(round(_ema_smooth(old[1], new[1], alpha))), 0),
        max(int(round(_ema_smooth(old[2], new[2], alpha))), 1),
        max(int(round(_ema_smooth(old[3], new[3], alpha))), 1),
    )


@dataclass
class PersistentTrack:
    target_id: str
    class_name: str
    state: str = "TENTATIVE"  # TENTATIVE, ACTIVE, LOST, REACQUIRED, STALE
    hits: int = 0
    missed_frames: int = 0
    last_bbox: tuple[int, int, int, int] = (0, 0, 1, 1)
    smoothed_bbox: tuple[int, int, int, int] = (0, 0, 1, 1)
    velocity: tuple[float, float] = (0.0, 0.0)
    predicted_bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    smoothed_confidence: float = 0.0
    confidence_history: list[float] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    reacquired_count: int = 0
    appearance: np.ndarray | None = None
    reacquire_cooldown: int = 0

    def __post_init__(self) -> None:
        if self.hits == 0:
            self.hits = 1
        if self.confidence > 0 and self.smoothed_confidence == 0:
            self.smoothed_confidence = self.confidence


class PersistentTrackManager:
    def __init__(
        self,
        iou_thresh: float = 0.2,
        center_dist_thresh: float = 100.0,
        max_lost_frames: int = 90,
        min_hits: int = 3,
        bbox_ema_alpha: float = 0.3,
        conf_ema_alpha: float = 0.3,
        reid_method: str = "colorhist",
        reid_threshold: float = 0.6,
        duplicate_iou: float = 0.5,
    ) -> None:
        self._iou_thresh = iou_thresh
        self._center_dist_thresh = center_dist_thresh
        self._max_lost_frames = max_lost_frames
        self._min_hits = min_hits
        self._bbox_ema_alpha = bbox_ema_alpha
        self._conf_ema_alpha = conf_ema_alpha
        self._reid_method = reid_method
        self._reid_threshold = reid_threshold
        self._duplicate_iou = duplicate_iou
        self._tracks: list[PersistentTrack] = []
        self._next_id = 1
        self._total_reacquisitions = 0
        self._total_id_switches = 0
        self._max_conf_history = 30

    def _make_id(self) -> str:
        tid = f"TGT-{self._next_id:03d}"
        self._next_id += 1
        return tid

    def _compute_appearance(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        if self._reid_method == "none":
            return None
        hist = compute_color_histogram(frame, bbox)
        return hist if np.linalg.norm(hist) > 1e-8 else None

    def _match_cost(
        self,
        det_bbox: tuple[int, int, int, int],
        det_class: str,
        det_appearance: np.ndarray | None,
        track: PersistentTrack,
    ) -> float:
        """Compute matching cost. Higher is better. Returns 0 if no match."""
        if det_class != track.class_name:
            return 0.0

        check_bbox = track.predicted_bbox or track.last_bbox
        iou = _bbox_iou(det_bbox, check_bbox)
        cdist = _center_dist(det_bbox, check_bbox)

        # Primary: IoU match
        if iou >= self._iou_thresh:
            cost = 0.5 + 0.3 * iou  # 0.5-0.8 range
        elif cdist <= self._center_dist_thresh:
            cost = 0.3 * (1.0 - cdist / self._center_dist_thresh)
        else:
            # Too far, but check appearance for reacquisition
            cost = 0.0

        # Boost with appearance similarity
        if det_appearance is not None and track.appearance is not None:
            sim = _cosine_similarity(det_appearance, track.appearance)
            if sim >= self._reid_threshold:
                cost = max(cost, 0.2 + 0.3 * sim)  # 0.2-0.5 range

        return cost

    def _greedy_match(
        self,
        dets: list[dict],
        det_appearances: list[np.ndarray | None],
        tracks: list[PersistentTrack],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy assignment. Returns (matched_pairs, unmatched_dets, unmatched_tracks)."""
        matched: list[tuple[int, int]] = []
        used_dets: set[int] = set()
        used_tracks: set[int] = set()

        for _ in range(min(len(dets), len(tracks))):
            best_cost = 0.1  # minimum threshold
            best_di = -1
            best_ti = -1
            for di in range(len(dets)):
                if di in used_dets:
                    continue
                for ti in range(len(tracks)):
                    if ti in used_tracks:
                        continue
                    cost = self._match_cost(
                        dets[di]["_bbox"],
                        dets[di]["class"],
                        det_appearances[di],
                        tracks[ti],
                    )
                    if cost > best_cost:
                        best_cost = cost
                        best_di = di
                        best_ti = ti
            if best_di < 0:
                break
            matched.append((best_di, best_ti))
            used_dets.add(best_di)
            used_tracks.add(best_ti)

        unmatched_dets = [i for i in range(len(dets)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    def _suppress_duplicates(self) -> None:
        """Remove tentative tracks that heavily overlap active tracks."""
        active = [t for t in self._tracks if t.state in ("ACTIVE", "REACQUIRED")]
        tentative = [t for t in self._tracks if t.state == "TENTATIVE"]
        to_remove: list[PersistentTrack] = []
        for ten in tentative:
            for act in active:
                if ten.class_name == act.class_name and _bbox_iou(ten.last_bbox, act.last_bbox) > self._duplicate_iou:
                    to_remove.append(ten)
                    break
        for t in to_remove:
            self._tracks.remove(t)

    def update(
        self,
        detections: list[dict],
        frame: np.ndarray,
        frame_id: int,
    ) -> list[dict]:
        """Process detections and return current track states as overlay dicts."""
        # Normalize detections
        for d in detections:
            d["_bbox"] = tuple(int(v) for v in d["bbox"])

        # Compute appearances
        appearances: list[np.ndarray | None] = []
        for d in detections:
            appearances.append(self._compute_appearance(frame, d["_bbox"]))

        # Separate tracks by state
        active_tracks = [t for t in self._tracks if t.state in ("ACTIVE", "REACQUIRED", "TENTATIVE")]
        lost_tracks = [t for t in self._tracks if t.state == "LOST"]

        # Step 1: Match detections to active/tentative/reacquired tracks
        matched, unmatched_det_idx, unmatched_track_idx = self._greedy_match(
            detections, appearances, active_tracks,
        )

        # Update matched tracks
        for di, ti in matched:
            det = detections[di]
            track = active_tracks[ti]
            new_bbox = det["_bbox"]
            old_center = _bbox_center(track.last_bbox)
            new_center = _bbox_center(new_bbox)
            track.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            track.last_bbox = new_bbox
            track.smoothed_bbox = _ema_bbox_smooth(track.smoothed_bbox, new_bbox, self._bbox_ema_alpha)
            track.confidence = det["confidence"]
            track.smoothed_confidence = _ema_smooth(track.smoothed_confidence, det["confidence"], self._conf_ema_alpha)
            track.confidence_history.append(det["confidence"])
            if len(track.confidence_history) > self._max_conf_history:
                track.confidence_history = track.confidence_history[-self._max_conf_history:]
            if appearances[di] is not None:
                if track.appearance is not None:
                    track.appearance = _ema_smooth(track.appearance, appearances[di], 0.2)
                    norm = np.linalg.norm(track.appearance)
                    if norm > 1e-8:
                        track.appearance = track.appearance / norm
                else:
                    track.appearance = appearances[di]
            track.hits += 1
            track.missed_frames = 0
            track.last_seen_frame = frame_id
            track.predicted_bbox = None
            if track.reacquire_cooldown > 0:
                track.reacquire_cooldown -= 1
            # Promote TENTATIVE → ACTIVE
            if track.state == "TENTATIVE" and track.hits >= self._min_hits:
                track.state = "ACTIVE"
            # REACQUIRED → ACTIVE after 1 hit
            if track.state == "REACQUIRED":
                track.state = "ACTIVE"

        # Step 2: Try to match unmatched detections to LOST tracks (reacquisition)
        unmatched_dets = [detections[i] for i in unmatched_det_idx]
        unmatched_det_apps = [appearances[i] for i in unmatched_det_idx]

        reacq_matched, still_unmatched_det, _ = self._greedy_match(
            unmatched_dets, unmatched_det_apps, lost_tracks,
        )

        new_unmatched_det_indices: list[int] = []
        for local_i, _ in reacq_matched:
            orig_idx = unmatched_det_idx[local_i]
            det = detections[orig_idx]
            track = lost_tracks[_]
            new_bbox = det["_bbox"]
            track.velocity = (0.0, 0.0)  # reset velocity after gap
            track.last_bbox = new_bbox
            track.smoothed_bbox = new_bbox  # reset smoothing after gap
            track.confidence = det["confidence"]
            track.smoothed_confidence = det["confidence"]
            track.hits += 1
            track.missed_frames = 0
            track.last_seen_frame = frame_id
            track.state = "REACQUIRED"
            track.reacquired_count += 1
            track.reacquire_cooldown = 10
            track.predicted_bbox = None
            if appearances[orig_idx] is not None:
                track.appearance = appearances[orig_idx]
            self._total_reacquisitions += 1

        for local_i in still_unmatched_det:
            new_unmatched_det_indices.append(unmatched_det_idx[local_i])

        # Step 3: Mark unmatched active tracks as LOST
        for ti in unmatched_track_idx:
            track = active_tracks[ti]
            if track.state in ("ACTIVE", "REACQUIRED"):
                track.state = "LOST"
            elif track.state == "TENTATIVE":
                track.missed_frames += 1
                if track.missed_frames > 3:
                    track.state = "STALE"

        # Step 4: Age all LOST tracks
        for t in self._tracks:
            if t.state == "LOST":
                t.missed_frames += 1
                if t.velocity != (0.0, 0.0):
                    t.predicted_bbox = _predict_bbox(t.last_bbox, t.velocity, t.missed_frames)
                if t.missed_frames > self._max_lost_frames:
                    t.state = "STALE"

        # Step 5: Create TENTATIVE tracks for genuinely new detections
        for idx in new_unmatched_det_indices:
            det = detections[idx]
            bbox = det["_bbox"]
            trk = PersistentTrack(
                target_id=self._make_id(),
                class_name=det["class"],
                state="TENTATIVE",
                hits=1,
                last_bbox=bbox,
                smoothed_bbox=bbox,
                confidence=det["confidence"],
                smoothed_confidence=det["confidence"],
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
                appearance=appearances[idx],
            )
            self._tracks.append(trk)

        # Step 6: Suppress duplicates
        self._suppress_duplicates()

        # Step 7: Clean up STALE tracks (keep last 100 for metrics)
        stale = [t for t in self._tracks if t.state == "STALE"]
        if len(stale) > 100:
            for t in stale[:-100]:
                self._tracks.remove(t)

        # Return all non-STALE tracks as overlay dicts
        return self._tracks_to_dets()

    def _tracks_to_dets(self) -> list[dict]:
        result = []
        for t in self._tracks:
            if t.state == "STALE":
                continue
            display_bbox = t.smoothed_bbox if t.state in ("ACTIVE", "REACQUIRED", "TENTATIVE") else (t.predicted_bbox or t.last_bbox)
            result.append({
                "class": t.class_name,
                "confidence": t.smoothed_confidence,
                "bbox": list(display_bbox),
                "track_id": t.target_id,
                "status": t.state,
                "age_frames": t.last_seen_frame - t.first_seen_frame + t.missed_frames,
                "missed_frames": t.missed_frames,
                "reacquired_count": t.reacquired_count,
                "hits": t.hits,
                "smoothed_bbox": True,
                "reacquire_cooldown": t.reacquire_cooldown,
            })
        return result

    def get_metrics(self) -> dict:
        active = sum(1 for t in self._tracks if t.state in ("ACTIVE", "REACQUIRED"))
        lost = sum(1 for t in self._tracks if t.state == "LOST")
        tentative = sum(1 for t in self._tracks if t.state == "TENTATIVE")
        stale = sum(1 for t in self._tracks if t.state == "STALE")
        confirmed = sum(1 for t in self._tracks if t.hits >= self._min_hits)
        return {
            "active": active,
            "lost": lost,
            "tentative": tentative,
            "stale": stale,
            "unique_confirmed": confirmed,
            "total_reacquisitions": self._total_reacquisitions,
            "total_tracks_created": self._next_id - 1,
        }
