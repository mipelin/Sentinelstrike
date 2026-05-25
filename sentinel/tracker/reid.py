"""Semantic ISR re-identification engine for aerial target tracking.

Provides multi-signal identity fingerprinting that combines:
- Appearance features (149-dim spatial-color from appearance.py)
- Spatial context (position, velocity, heading prediction)
- Temporal history (track age, hit count, appearance stability)

Designed for UAV ISR: camera motion, vegetation occlusion, varying scale.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .appearance import AppearanceGallery, cosine_distance, extract_identity_descriptor


@dataclass
class SimilarityBreakdown:
    """Individual signal scores from a multi-signal comparison."""
    appearance: float = 0.0
    spatial: float = 0.0
    heading: float = 0.0
    combined: float = 0.0


class SemanticIdentity:
    """Multi-signal identity fingerprint for ISR re-identification.

    Wraps appearance features, spatial context, and temporal history
    into a single identity record that survives track lifecycle changes.
    """

    def __init__(self, track_id: str, class_name: str) -> None:
        self.track_id = track_id
        self.class_name = class_name

        self._gallery = AppearanceGallery(max_size=15)
        self._descriptor: dict = {}
        self._last_update_frame = -1

        self._first_frame = 0
        self._hit_count = 0

        self._positions: list[tuple[float, float]] = []
        self._last_cx_norm: float | None = None
        self._last_cy_norm: float | None = None
        self._velocity: tuple[float, float] = (0.0, 0.0)
        self._heading: float = 0.0
        self._speed: float = 0.0
        self._frame_w: int = 1280
        self._frame_h: int = 720

        self._archive_time: float = 0.0

    def update_appearance(
        self,
        feature: np.ndarray | None,
        frame: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        frame_id: int,
    ) -> None:
        if frame_id <= self._last_update_frame:
            return
        self._last_update_frame = frame_id
        if feature is not None:
            self._gallery.update(feature)
        if frame is not None:
            self._descriptor = extract_identity_descriptor(frame, bbox)

    def update_spatial(
        self,
        cx: float,
        cy: float,
        velocity: tuple[float, float],
        heading: float,
        speed: float,
        frame_w: int,
        frame_h: int,
        frame_id: int,
    ) -> None:
        self._hit_count += 1
        if self._first_frame == 0:
            self._first_frame = frame_id

        self._frame_w = max(frame_w, 1)
        self._frame_h = max(frame_h, 1)
        cx_n = cx / self._frame_w
        cy_n = cy / self._frame_h
        self._positions.append((cx_n, cy_n))
        if len(self._positions) > 30:
            self._positions = self._positions[-30:]
        self._last_cx_norm = cx_n
        self._last_cy_norm = cy_n
        self._velocity = velocity
        self._heading = heading
        self._speed = speed

    @property
    def gallery(self) -> AppearanceGallery:
        return self._gallery

    @property
    def descriptor(self) -> dict:
        return self._descriptor

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def track_age_frames(self) -> int:
        return self._hit_count

    @property
    def appearance_stability(self) -> float:
        return self._gallery.stability_score()

    @property
    def last_position(self) -> tuple[float, float] | None:
        if self._last_cx_norm is None:
            return None
        return (self._last_cx_norm * self._frame_w, self._last_cy_norm * self._frame_h)

    @property
    def last_velocity(self) -> tuple[float, float]:
        return self._velocity

    @property
    def last_heading(self) -> float:
        return self._heading

    @property
    def last_speed(self) -> float:
        return self._speed

    def appearance_similarity(self, other_feature: np.ndarray | None) -> float:
        if other_feature is None:
            return 0.0
        dist = self._gallery.match_distance(other_feature)
        return max(0.0, 1.0 - dist)

    def predict_position(self, frames_ahead: int) -> tuple[float, float] | None:
        if self._last_cx_norm is None:
            return None
        decay = 0.95 ** frames_ahead
        vx_norm = self._velocity[0] / self._frame_w
        vy_norm = self._velocity[1] / self._frame_h
        pred_cx = (self._last_cx_norm + vx_norm * frames_ahead * decay) * self._frame_w
        pred_cy = (self._last_cy_norm + vy_norm * frames_ahead * decay) * self._frame_h
        return (pred_cx, pred_cy)

    def spatial_distance_score(
        self,
        other_cx: float,
        other_cy: float,
        frame_w: int,
        frame_h: int,
    ) -> float:
        if self._last_cx_norm is None:
            return 0.5
        fw = max(frame_w, 1)
        fh = max(frame_h, 1)
        pred = self.last_position
        if pred is None:
            return 0.5
        dx = other_cx - pred[0]
        dy = other_cy - pred[1]
        dist = math.sqrt(dx * dx + dy * dy)
        sigma = 0.15 * math.sqrt(fw * fw + fh * fh)
        return math.exp(-dist / sigma)

    def heading_consistency(self, other_heading: float) -> float:
        if self._speed < 0.5:
            return 0.5
        cos_sim = math.cos(self._heading - other_heading)
        return 0.5 + 0.5 * cos_sim

    def compute_similarity(
        self,
        other_feature: np.ndarray | None,
        other_cx: float,
        other_cy: float,
        other_heading: float,
        frame_w: int,
        frame_h: int,
        *,
        appearance_weight: float = 0.5,
        spatial_weight: float = 0.3,
        heading_weight: float = 0.2,
    ) -> tuple[float, SimilarityBreakdown]:
        app_sim = self.appearance_similarity(other_feature)
        spa_sim = self.spatial_distance_score(other_cx, other_cy, frame_w, frame_h)
        hdg_sim = self.heading_consistency(other_heading)

        aw = appearance_weight
        sw = spatial_weight
        hw = heading_weight

        if self.appearance_stability > 0.8:
            aw *= 1.3
            sw *= 0.8
        if self._speed > 2.0:
            hw *= 1.5
        if self._hit_count > 50:
            app_sim = min(1.0, app_sim + 0.05)

        total = aw + sw + hw
        combined = (aw * app_sim + sw * spa_sim + hw * hdg_sim) / total

        breakdown = SimilarityBreakdown(
            appearance=round(app_sim, 3),
            spatial=round(spa_sim, 3),
            heading=round(hdg_sim, 3),
            combined=round(combined, 3),
        )
        return combined, breakdown

    def mark_archived(self) -> None:
        self._archive_time = time.monotonic()

    @property
    def archive_age_s(self) -> float:
        if self._archive_time == 0.0:
            return 0.0
        return time.monotonic() - self._archive_time


class IdentityMemory:
    """Survives track deletion for cross-lifecycle Re-ID.

    Archives SemanticIdentity objects when tracks are deleted, allowing
    new detections to be matched against previously seen identities.
    """

    def __init__(
        self,
        max_entries: int = 50,
        timeout_seconds: float = 30.0,
        min_archive_hits: int = 3,
    ) -> None:
        self._entries: list[SemanticIdentity] = []
        self._max_entries = max_entries
        self._timeout = timeout_seconds
        self._min_hits = min_archive_hits

    def archive(self, identity: SemanticIdentity) -> None:
        if identity.hit_count < self._min_hits:
            return
        identity.mark_archived()
        self._entries.append(identity)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def find_match(
        self,
        feature: np.ndarray | None,
        cx: float,
        cy: float,
        heading: float,
        frame_w: int,
        frame_h: int,
        class_name: str,
        threshold: float = 0.6,
    ) -> tuple[str | None, float, SimilarityBreakdown]:
        self.purge_expired(time.monotonic())
        best_id = None
        best_score = 0.0
        best_breakdown = SimilarityBreakdown()
        for entry in self._entries:
            if entry.class_name != class_name:
                continue
            pred = entry.predict_position(
                int(entry.archive_age_s * 15)  # assume 15 fps default
            )
            if pred is not None:
                dx = cx - pred[0]
                dy = cy - pred[1]
                dist = math.sqrt(dx * dx + dy * dy)
                sigma = 0.3 * math.sqrt(frame_w ** 2 + frame_h ** 2)
                if dist > 2.0 * sigma:
                    continue
            score, breakdown = entry.compute_similarity(
                feature, cx, cy, heading, frame_w, frame_h,
            )
            if score > best_score:
                best_score = score
                best_id = entry.track_id
                best_breakdown = breakdown
        if best_score < threshold:
            return None, best_score, best_breakdown
        return best_id, best_score, best_breakdown

    def purge_expired(self, current_time: float) -> int:
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if (current_time - e._archive_time) < self._timeout
        ]
        return before - len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)


@dataclass
class ReIDMetrics:
    """Tracks Re-ID performance metrics over a mission."""
    id_switches: int = 0
    reacquisitions: int = 0
    false_reacquisitions: int = 0
    cross_lifecycle_matches: int = 0
    _identity_starts: dict = field(default_factory=dict)
    _events: list = field(default_factory=list)

    def record_id_switch(
        self, old_id: str, new_id: str, frame_id: int, similarity: float,
    ) -> None:
        self.id_switches += 1
        self._events.append({
            "type": "id_switch", "old_id": old_id, "new_id": new_id,
            "frame": frame_id, "similarity": round(similarity, 3),
        })

    def record_reacquisition(
        self, track_id: str, frame_id: int, lost_frames: int, similarity: float,
    ) -> None:
        self.reacquisitions += 1
        self._events.append({
            "type": "reacquisition", "track_id": track_id,
            "frame": frame_id, "lost_frames": lost_frames,
            "similarity": round(similarity, 3),
        })

    def record_cross_lifecycle_match(
        self, original_id: str, new_id: str, frame_id: int, similarity: float,
    ) -> None:
        self.cross_lifecycle_matches += 1
        self._events.append({
            "type": "cross_lifecycle", "original_id": original_id,
            "new_id": new_id, "frame": frame_id,
            "similarity": round(similarity, 3),
        })

    def record_identity_start(self, track_id: str, frame_id: int) -> None:
        self._identity_starts[track_id] = frame_id

    def record_identity_end(self, track_id: str, frame_id: int) -> None:
        pass

    def get_summary(self) -> dict:
        return {
            "id_switches": self.id_switches,
            "reacquisitions": self.reacquisitions,
            "false_reacquisitions": self.false_reacquisitions,
            "cross_lifecycle_matches": self.cross_lifecycle_matches,
            "total_identities": len(self._identity_starts),
            "recent_events": self._events[-20:],
        }
