"""ISR target priority scoring.

Computes interest scores based on behavioral analysis, persistence,
and operator overrides to recommend which targets deserve attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .behavior import BehaviorAnalyzer


@dataclass
class PriorityWeights:
    concealment_behavior: float = 0.25
    anomaly_score: float = 0.25
    persistence: float = 0.20
    interaction_complexity: float = 0.15
    operator_priority: float = 0.15


class PriorityScorer:
    """Computes ISR interest priority for tracked targets."""

    def __init__(
        self,
        behavior_analyzer: BehaviorAnalyzer,
        weights: PriorityWeights | None = None,
        operator_overrides: dict[str, float] | None = None,
    ) -> None:
        self._behavior = behavior_analyzer
        self._weights = weights or PriorityWeights()
        self._operator_overrides = operator_overrides or {}
        self._persistence_scale: int = 300

    def score_target(self, track: dict) -> float:
        """Compute priority score for a single target. Returns 0.0-1.0."""
        tid = track.get("track_id", "")
        profile = self._behavior.get_profile(tid)

        w = self._weights
        score = 0.0

        if profile:
            score += w.concealment_behavior * profile.concealment_affinity
            score += w.anomaly_score * profile.anomaly_score
            score += w.interaction_complexity * min(profile.interaction_count / 3.0, 1.0)

        hits = track.get("hits", 0)
        score += w.persistence * min(hits / self._persistence_scale, 1.0)

        score += w.operator_priority * self._operator_overrides.get(tid, 0.0)

        return max(0.0, min(1.0, score))

    def rank_targets(self, tracks: list[dict]) -> list[tuple[str, float]]:
        """Score and rank all targets. Returns [(track_id, score)], highest first."""
        scored = [(t.get("track_id", ""), self.score_target(t)) for t in tracks if t.get("track_id")]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def suggest_target(self, tracks: list[dict]) -> str | None:
        """Return track_id of highest-priority visible target."""
        visible = [t for t in tracks if t.get("occlusion_state", "") == "VISIBLE"]
        if not visible:
            return None
        ranked = self.rank_targets(visible)
        return ranked[0][0] if ranked else None

    def set_operator_override(self, track_id: str, priority: float) -> None:
        self._operator_overrides[track_id] = max(0.0, min(1.0, priority))
