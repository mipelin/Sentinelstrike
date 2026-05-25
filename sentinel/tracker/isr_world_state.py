"""ISR world-state layer — unified query interface for target awareness.

Wraps HypothesisManager + GeospatialLayer + BehaviorAnalyzer into a single
queryable object. Provides uncertainty regions, search priorities, behavioral
profiles, and activity maps for follow controllers and overlay rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .behavior import BehaviorAnalyzer
    from .geospatial import GeospatialLayer
    from .hypothesis import HypothesisManager
    from .priority import PriorityScorer
    from .terrain_reasoning import TerrainReasoner


@dataclass
class UncertaintyRegion:
    """Region where a target might be, derived from hypothesis spread."""
    track_id: str
    center_lat: float
    center_lon: float
    radius_m: float
    confidence: float
    sources: list[str] = field(default_factory=list)


@dataclass
class SearchPriority:
    """Weighted search direction for a lost target."""
    track_id: str
    bearing_deg: float
    width_deg: float
    probability: float
    source_type: str
    predicted_lat: float
    predicted_lon: float
    search_radius_m: float


class ISRWorldState:
    """Unified ISR world-awareness query interface."""

    def __init__(
        self,
        terrain: TerrainReasoner,
        geospatial: GeospatialLayer,
    ) -> None:
        self._terrain = terrain
        self._geospatial = geospatial
        self._hypothesis_mgr: HypothesisManager | None = None
        self._behavior: BehaviorAnalyzer | None = None
        self._priority_scorer: PriorityScorer | None = None

    def attach_hypothesis_manager(self, mgr: HypothesisManager) -> None:
        self._hypothesis_mgr = mgr

    def attach_behavior(self, behavior: BehaviorAnalyzer) -> None:
        self._behavior = behavior

    def attach_priority_scorer(self, scorer: PriorityScorer) -> None:
        self._priority_scorer = scorer

    def update(self, tracks: list[dict]) -> None:
        """Update geospatial layer from tracker output."""
        interaction_pairs = []
        if self._hypothesis_mgr:
            interaction_pairs = [
                list(pair) for pair in self._hypothesis_mgr.get_merged_groups()
            ]
        self._geospatial.update(tracks, interaction_pairs=interaction_pairs or None)

    def get_hypotheses(self, track_id: str) -> list[dict]:
        if not self._hypothesis_mgr:
            return []
        return [
            {"type": h.htype.value, "probability": round(h.probability, 3),
             "predicted_lat": round(h.predicted_lat, 6),
             "predicted_lon": round(h.predicted_lon, 6),
             "search_radius_m": round(h.search_radius_m, 1)}
            for h in self._hypothesis_mgr.get_hypotheses(track_id)
        ]

    def get_uncertainty_regions(self) -> list[UncertaintyRegion]:
        """Compute uncertainty regions from hypothesis spread."""
        if not self._hypothesis_mgr:
            return []
        regions = []
        for tid, hyps in self._hypothesis_mgr._hypotheses.items():
            if not hyps:
                continue
            total_w = sum(h.probability for h in hyps)
            if total_w < 0.01:
                continue
            lat = sum(h.predicted_lat * h.probability for h in hyps) / total_w
            lon = sum(h.predicted_lon * h.probability for h in hyps) / total_w
            max_dist = 0.0
            for h in hyps:
                dn = (h.predicted_lat - lat) * 111320.0
                de = (h.predicted_lon - lon) * 111320.0 * math.cos(math.radians(lat))
                dist = math.sqrt(dn * dn + de * de)
                max_dist = max(max_dist, dist)
            radius = max(max_dist, 5.0) + max(h.search_radius_m for h in hyps) * 0.5
            best_prob = max(h.probability for h in hyps)
            sources = list({h.htype.value for h in hyps})
            regions.append(UncertaintyRegion(
                track_id=tid,
                center_lat=lat,
                center_lon=lon,
                radius_m=round(radius, 1),
                confidence=round(best_prob, 3),
                sources=sources,
            ))
        return regions

    def get_search_priorities(self, follow_id: str | None = None) -> list[SearchPriority]:
        """Get search priority sectors, optionally filtered to one target."""
        if not self._hypothesis_mgr:
            return []
        target_ids = [follow_id] if follow_id else list(self._hypothesis_mgr._hypotheses.keys())
        priorities = []
        for tid in target_ids:
            sectors = self._hypothesis_mgr.get_search_sectors(tid)
            for sec in sectors:
                priorities.append(SearchPriority(
                    track_id=tid,
                    bearing_deg=sec["bearing_deg"],
                    width_deg=sec["width_deg"],
                    probability=sec["probability"],
                    source_type=sec["source_type"],
                    predicted_lat=sec["predicted_lat"],
                    predicted_lon=sec["predicted_lon"],
                    search_radius_m=sec["search_radius_m"],
                ))
        priorities.sort(key=lambda p: p.probability, reverse=True)
        return priorities[:10]

    def get_activity_map(self) -> list[dict]:
        return self._geospatial.get_heatmap()

    def get_last_seen(self, track_id: str) -> tuple[float, float] | None:
        return self._geospatial.get_last_seen(track_id)

    def get_all_last_seen(self) -> dict[str, tuple[float, float, float]]:
        return self._geospatial.get_all_last_seen()

    def get_target_trail(self, track_id: str, max_points: int = 200) -> list[tuple[float, float]]:
        return self._geospatial.get_trail(track_id, max_points)

    # --- Behavioral queries ---

    def get_behavioral_profiles(self) -> dict:
        """Get all behavioral profiles as serializable dicts."""
        if not self._behavior:
            return {}
        profiles = {}
        for tid, p in self._behavior.get_all_profiles().items():
            profiles[tid] = {
                "track_id": p.track_id,
                "class_name": p.class_name,
                "pattern": p.current_pattern.value,
                "avg_speed_mps": round(p.avg_speed_mps, 2),
                "concealment_affinity": round(p.concealment_affinity, 3),
                "anomaly_score": round(p.anomaly_score, 3),
                "interaction_count": p.interaction_count,
                "stop_count": p.stop_count,
                "direction_changes": p.direction_changes,
            }
        return profiles

    def get_targets_by_pattern(self, pattern: str) -> list[str]:
        """Get track IDs matching a movement pattern."""
        if not self._behavior:
            return []
        return [
            tid for tid, p in self._behavior.get_all_profiles().items()
            if p.current_pattern.value == pattern
        ]

    def get_high_anomaly_targets(self, threshold: float = 0.7) -> list[dict]:
        """Get targets with anomaly score above threshold."""
        if not self._behavior:
            return []
        return [
            {"track_id": tid, "anomaly_score": round(p.anomaly_score, 3),
             "pattern": p.current_pattern.value}
            for tid, p in self._behavior.get_all_profiles().items()
            if p.anomaly_score >= threshold
        ]

    def get_interaction_summary(self) -> list[dict]:
        """Get current target interactions."""
        if not self._behavior:
            return []
        result = []
        for (id_a, id_b), rec in self._behavior.get_interaction_graph().items():
            result.append({
                "id_a": id_a,
                "id_b": id_b,
                "count": rec.count,
                "duration_frames": rec.duration_frames,
                "type": rec.interaction_type,
            })
        return result

    def get_priority_ranking(self, tracks: list[dict]) -> list[tuple[str, float]]:
        """Score and rank targets by priority."""
        if not self._priority_scorer:
            return []
        return self._priority_scorer.rank_targets(tracks)

    def get_behavioral_heatmaps(self) -> dict:
        """Get all behavioral heatmaps."""
        return {
            "loitering": self._geospatial.get_loitering_zones(),
            "concealment": self._geospatial.get_concealment_hotspots(),
            "interaction": self._geospatial.get_interaction_zones(),
        }

    @property
    def target_count(self) -> int:
        return self._geospatial.target_count
