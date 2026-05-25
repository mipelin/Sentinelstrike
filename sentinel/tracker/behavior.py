"""Lightweight behavioral ISR reasoning for target analysis.

Classifies movement patterns, detects concealment-seeking behavior,
tracks interactions between targets, and computes anomaly scores.
All derived from existing ISR tracker data — no new detection needed.

Uses simple heuristics and EMA smoothing, not heavy ML models.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_reasoning import TerrainReasoner


class MovementPattern(str, Enum):
    NORMAL_WALKING = "NORMAL_WALKING"
    LOITERING = "LOITERING"
    EVASIVE = "EVASIVE"
    STOP_AND_GO = "STOP_AND_GO"
    CONCEALMENT_SEEKING = "CONCEALMENT_SEEKING"
    ROAD_FOLLOWING = "ROAD_FOLLOWING"
    GROUP_MOVEMENT = "GROUP_MOVEMENT"
    ERRATIC = "ERRATIC"
    STATIONARY_OBSERVATION = "STATIONARY_OBSERVATION"


@dataclass
class BehavioralProfile:
    track_id: str
    class_name: str = ""
    avg_speed_mps: float = 0.0
    stop_count: int = 0
    direction_changes: int = 0
    time_stationary_s: float = 0.0
    time_hidden_s: float = 0.0
    total_frames: int = 0
    terrain_counts: dict[str, int] = field(default_factory=lambda: {"open": 0, "vegetation": 0, "road": 0})
    concealment_affinity: float = 0.0
    path_repetition: float = 0.0
    anomaly_score: float = 0.0
    current_pattern: MovementPattern = MovementPattern.NORMAL_WALKING
    interaction_count: int = 0

    # Internal state
    _was_stationary: bool = False
    _stationary_frames: int = 0
    _prev_heading: float = 0.0
    _speed_ema_alpha: float = 0.1
    _pattern_confidence: dict[str, float] = field(default_factory=dict)
    _visited_cells: set[tuple[int, int]] = field(default_factory=set)
    _cell_revisits: int = 0


@dataclass
class InteractionRecord:
    id_a: str
    id_b: str
    count: int = 0
    duration_frames: int = 0
    interaction_type: str = "proximity"
    last_seen_frame: int = 0


@dataclass
class BehaviorMetrics:
    pattern_counts: dict[str, int] = field(default_factory=dict)
    avg_anomaly_score: float = 0.0
    total_interactions: int = 0
    concealment_events: int = 0
    target_avg_lifetime_frames: float = 0.0

    def update(self, profiles: dict[str, BehavioralProfile]) -> dict:
        if not profiles:
            return self.get_summary()
        self.pattern_counts = Counter(
            p.current_pattern.value for p in profiles.values()
        )
        scores = [p.anomaly_score for p in profiles.values()]
        self.avg_anomaly_score = sum(scores) / len(scores) if scores else 0.0
        self.concealment_events = sum(
            1 for p in profiles.values() if p.concealment_affinity > 0.5
        )
        lifetimes = [p.total_frames for p in profiles.values()]
        self.target_avg_lifetime_frames = (
            sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
        )
        return self.get_summary()

    def get_summary(self) -> dict:
        return {
            "pattern_counts": dict(self.pattern_counts),
            "avg_anomaly_score": round(self.avg_anomaly_score, 3),
            "total_interactions": self.total_interactions,
            "concealment_events": self.concealment_events,
            "target_avg_lifetime_frames": round(self.target_avg_lifetime_frames, 1),
        }


_STOP_THRESHOLD_MPS = 0.3
_STOP_MIN_FRAMES = 5
_HEADING_CHANGE_DEG = 30.0
_CELL_SIZE_M = 3.0
_PATTERN_EMA_ALPHA = 0.15


class BehaviorAnalyzer:
    """Maintains behavioral profiles for all tracked targets."""

    def __init__(
        self,
        terrain: TerrainReasoner,
        fps: float = 15.0,
        interaction_distance_m: float = 3.0,
        interaction_min_frames: int = 10,
    ) -> None:
        self._terrain = terrain
        self._fps = fps
        self._interaction_dist = interaction_distance_m
        self._interaction_min = interaction_min_frames
        self._profiles: dict[str, BehavioralProfile] = {}
        self._interactions: dict[tuple[str, str], InteractionRecord] = {}
        self._frame_count: int = 0
        self.metrics = BehaviorMetrics()

    def update(self, tracks: list[dict]) -> None:
        """Update all profiles and interactions from tracker output."""
        self._frame_count += 1

        for t in tracks:
            wp = t.get("world_position")
            if not wp or not wp.get("lat"):
                continue
            self._update_profile(t)

        self._update_interactions(tracks)
        self._compute_anomaly_scores()
        self.metrics.update(self._profiles)

    def get_profile(self, track_id: str) -> BehavioralProfile | None:
        return self._profiles.get(track_id)

    def get_all_profiles(self) -> dict[str, BehavioralProfile]:
        return dict(self._profiles)

    def get_interaction_graph(self) -> dict[tuple[str, str], InteractionRecord]:
        return dict(self._interactions)

    def remove_profile(self, track_id: str) -> None:
        self._profiles.pop(track_id, None)
        to_remove = [k for k in self._interactions if track_id in k]
        for k in to_remove:
            del self._interactions[k]

    # --- Internal ---

    def _update_profile(self, track: dict) -> BehavioralProfile:
        tid = track.get("track_id", "")
        if not tid:
            return None

        if tid not in self._profiles:
            self._profiles[tid] = BehavioralProfile(
                track_id=tid,
                class_name=track.get("class", ""),
            )
        p = self._profiles[tid]

        p.total_frames += 1
        wv = track.get("world_velocity", {})
        speed_mps = wv.get("speed_mps", 0) if wv else 0
        heading_rad = wv.get("heading_rad", 0) if wv else 0

        # Speed EMA
        alpha = p._speed_ema_alpha
        p.avg_speed_mps = alpha * speed_mps + (1 - alpha) * p.avg_speed_mps

        # Stop detection
        if speed_mps < _STOP_THRESHOLD_MPS:
            p._stationary_frames += 1
            if not p._was_stationary and p._stationary_frames >= _STOP_MIN_FRAMES:
                p.stop_count += 1
                p._was_stationary = True
            p.time_stationary_s += 1.0 / self._fps
        else:
            p._stationary_frames = 0
            p._was_stationary = False

        # Heading change detection
        if p.total_frames > 1:
            diff = abs(heading_rad - p._prev_heading)
            diff = min(diff, 2 * math.pi - diff)
            if math.degrees(diff) > _HEADING_CHANGE_DEG:
                p.direction_changes += 1
        p._prev_heading = heading_rad

        # Hidden time
        occ = track.get("occlusion_state", "VISIBLE")
        if occ != "VISIBLE":
            p.time_hidden_s += 1.0 / self._fps

        # Terrain classification
        wp = track.get("world_position", {})
        lat, lon = wp.get("lat", 0), wp.get("lon", 0)
        if lat and self._terrain.zone_count > 0:
            terrain_type = self._terrain.terrain_type_at(lat, lon)
            p.terrain_counts[terrain_type] = p.terrain_counts.get(terrain_type, 0) + 1

        # Concealment affinity
        total_terrain = sum(p.terrain_counts.values())
        if total_terrain > 10:
            p.concealment_affinity = p.terrain_counts.get("vegetation", 0) / total_terrain

        # Path repetition (grid cell revisits)
        cell = self._to_cell(lat, lon)
        if cell in p._visited_cells:
            p._cell_revisits += 1
        p._visited_cells.add(cell)
        if len(p._visited_cells) > 0:
            p.path_repetition = min(1.0, p._cell_revisits / max(len(p._visited_cells) * 2, 1))

        # Pattern classification
        p.current_pattern = self._classify_pattern(p)

        # Interaction count
        p.interaction_count = sum(
            1 for k in self._interactions if tid in k
        )

        return p

    def _classify_pattern(self, p: BehavioralProfile) -> MovementPattern:
        total = max(p.total_frames, 1)
        road_ratio = p.terrain_counts.get("road", 0) / total
        speed = p.avg_speed_mps
        stops = p.stop_count
        dir_ch = p.direction_changes
        conceal = p.concealment_affinity

        # Score each pattern
        scores: dict[str, float] = {}

        if speed < 0.3 and total > 60 and stops <= 1:
            scores["STATIONARY_OBSERVATION"] = 0.9
        elif speed < 0.3 and total > 30:
            scores["STATIONARY_OBSERVATION"] = 0.5

        if stops >= 3 and speed < 1.0 and dir_ch >= 3:
            scores["LOITERING"] = 0.9
        elif stops >= 2 and speed < 1.0:
            scores["LOITERING"] = 0.5

        if dir_ch >= 5 and speed > 1.5 and conceal > 0.4:
            scores["EVASIVE"] = 0.9
        elif dir_ch >= 4 and conceal > 0.3:
            scores["EVASIVE"] = 0.5

        if stops >= 3 and speed > 0.5:
            scores["STOP_AND_GO"] = 0.8

        if conceal > 0.5 and p.terrain_counts.get("vegetation", 0) > p.terrain_counts.get("open", 0):
            scores["CONCEALMENT_SEEKING"] = 0.9
        elif conceal > 0.3:
            scores["CONCEALMENT_SEEKING"] = 0.4

        if road_ratio > 0.7 and speed > 0.5:
            scores["ROAD_FOLLOWING"] = 0.8

        if p.interaction_count >= 1 and speed > 0.3:
            scores["GROUP_MOVEMENT"] = 0.7

        if dir_ch >= 8 and speed > 1.0:
            scores["ERRATIC"] = 0.9
        elif dir_ch >= 5:
            scores["ERRATIC"] = 0.4

        # EMA smoothing
        for k, v in scores.items():
            old = p._pattern_confidence.get(k, 0.0)
            p._pattern_confidence[k] = _PATTERN_EMA_ALPHA * v + (1 - _PATTERN_EMA_ALPHA) * old

        # Decay absent patterns
        for k in list(p._pattern_confidence.keys()):
            if k not in scores:
                p._pattern_confidence[k] *= (1 - _PATTERN_EMA_ALPHA)

        # Pick best
        if p._pattern_confidence:
            best = max(p._pattern_confidence, key=lambda k: p._pattern_confidence[k])
            if p._pattern_confidence[best] > 0.3:
                try:
                    return MovementPattern(best)
                except ValueError:
                    pass

        return MovementPattern.NORMAL_WALKING

    def _update_interactions(self, tracks: list[dict]) -> None:
        """Check pairwise proximity between targets."""
        active = []
        for t in tracks:
            wp = t.get("world_position")
            if wp and wp.get("lat"):
                active.append(t)

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                wa, wb = a["world_position"], b["world_position"]
                dist = _haversine_m(
                    wa["lat"], wa["lon"], wb["lat"], wb["lon"],
                )
                if dist < self._interaction_dist:
                    key = _pair_key(a["track_id"], b["track_id"])
                    if key not in self._interactions:
                        itype = _classify_interaction(a, b)
                        self._interactions[key] = InteractionRecord(
                            id_a=key[0], id_b=key[1], interaction_type=itype,
                        )
                    rec = self._interactions[key]
                    rec.count += 1
                    rec.duration_frames += 1
                    rec.last_seen_frame = self._frame_count
                    self.metrics.total_interactions += 1

        # Decay old interactions (older than 30s)
        threshold = int(self._fps * 30)
        expired = [
            k for k, v in self._interactions.items()
            if self._frame_count - v.last_seen_frame > threshold
        ]
        for k in expired:
            del self._interactions[k]

    def _compute_anomaly_scores(self) -> None:
        """Population-relative anomaly scoring."""
        if len(self._profiles) < 2:
            return

        speeds = [p.avg_speed_mps for p in self._profiles.values()]
        stops = [p.stop_count / max(p.total_frames, 1) for p in self._profiles.values()]
        conceals = [p.concealment_affinity for p in self._profiles.values()]
        headings = [p.direction_changes / max(p.total_frames, 1) for p in self._profiles.values()]

        def _zscore(val: float, values: list[float]) -> float:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            if std < 1e-6:
                return 0.0
            return min(abs(val - mean) / std, 3.0) / 3.0

        for p in self._profiles.values():
            total = max(p.total_frames, 1)
            s_z = _zscore(p.avg_speed_mps, speeds)
            st_z = _zscore(p.stop_count / total, stops)
            c_z = _zscore(p.concealment_affinity, conceals)
            h_z = _zscore(p.direction_changes / total, headings)
            p.anomaly_score = min(1.0, 0.3 * s_z + 0.2 * st_z + 0.3 * c_z + 0.2 * h_z)

    @staticmethod
    def _to_cell(lat: float, lon: float) -> tuple[int, int]:
        return (
            int(lat * 111320.0 / _CELL_SIZE_M),
            int(lon * 111320.0 / _CELL_SIZE_M),
        )


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _classify_interaction(a: dict, b: dict) -> str:
    ca, cb = a.get("class", ""), b.get("class", "")
    if ("car" in ca or "truck" in ca) and "person" in cb:
        return "vehicle_stop"
    if ("car" in cb or "truck" in cb) and "person" in ca:
        return "vehicle_stop"
    wa = a.get("world_velocity", {}) or {}
    wb = b.get("world_velocity", {}) or {}
    ha = wa.get("heading_rad", 0)
    hb = wb.get("heading_rad", 0)
    if ha and hb:
        diff = abs(ha - hb)
        diff = min(diff, 2 * math.pi - diff)
        if math.degrees(diff) < 30:
            return "following"
    return "proximity"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
