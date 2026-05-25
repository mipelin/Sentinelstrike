"""Lightweight multi-hypothesis ISR target tracking.

Instead of committing to a single "target disappeared" interpretation, maintains
up to 5 probabilistic hypotheses per target explaining what happened and where
the target likely is. Hypothesis probabilities evolve over time using terrain
context, movement history, and appearance consistency.

NOT academic MHT — hard cap of 5 hypotheses per target, simple scalar decay,
no Bayesian belief propagation. Designed for real-time ISR at <0.5ms overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_reasoning import TerrainReasoner
    from .world_projection import DronePose

from .world_projection import latlon_to_local, local_to_latlon


MAX_HYPOTHESES = 5
MIN_PROB = 0.05


class HypothesisType(str, Enum):
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    CONTINUED_PATH = "CONTINUED_PATH"
    EXITED_FOV = "EXITED_FOV"
    MERGED_GROUP = "MERGED_GROUP"
    STATIONARY = "STATIONARY"
    FALSE_MATCH = "FALSE_MATCH"
    REAPPEARING = "REAPPEARING"


@dataclass
class TargetHypothesis:
    htype: HypothesisType
    predicted_lat: float
    predicted_lon: float
    velocity_mps: float
    heading_rad: float
    probability: float
    occlusion_prob: float = 0.0
    appearance_consistency: float = 0.0
    age_frames: int = 0
    search_radius_m: float = 5.0


@dataclass
class HypothesisMetrics:
    occlusion_recovery_correct: int = 0
    occlusion_recovery_total: int = 0
    merge_recovery_correct: int = 0
    merge_recovery_total: int = 0
    false_reacquisition: int = 0
    total_resolved: int = 0
    _hypothesis_count_samples: list[float] = field(default_factory=list)

    def record_recovery(self, winning_type: str, correct: bool) -> None:
        self.total_resolved += 1
        if winning_type == "OCCLUDED":
            self.occlusion_recovery_total += 1
            if correct:
                self.occlusion_recovery_correct += 1
        elif winning_type == "MERGED_GROUP":
            self.merge_recovery_total += 1
            if correct:
                self.merge_recovery_correct += 1
        if not correct:
            self.false_reacquisition += 1

    def record_hypothesis_count(self, count: float) -> None:
        self._hypothesis_count_samples.append(count)

    def get_summary(self) -> dict:
        occ_rate = (
            self.occlusion_recovery_correct / self.occlusion_recovery_total
            if self.occlusion_recovery_total > 0 else 0.0
        )
        merge_rate = (
            self.merge_recovery_correct / self.merge_recovery_total
            if self.merge_recovery_total > 0 else 0.0
        )
        false_rate = (
            self.false_reacquisition / self.total_resolved
            if self.total_resolved > 0 else 0.0
        )
        avg_count = (
            sum(self._hypothesis_count_samples) / len(self._hypothesis_count_samples)
            if self._hypothesis_count_samples else 0.0
        )
        return {
            "occlusion_recovery_rate": round(occ_rate, 3),
            "merge_recovery_rate": round(merge_rate, 3),
            "false_reacquisition_rate": round(false_rate, 3),
            "total_resolved": self.total_resolved,
            "avg_hypothesis_count": round(avg_count, 2),
        }


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class HypothesisManager:
    """Manages multi-hypothesis state for all tracked targets."""

    def __init__(self, terrain: TerrainReasoner, fps: float = 15.0) -> None:
        self._terrain = terrain
        self._fps = fps
        self._hypotheses: dict[str, list[TargetHypothesis]] = {}
        self._merge_pairs: dict[str, list[str]] = {}
        self._last_visible_state: dict[str, str] = {}
        self._behavior_hints: dict[str, dict] = {}
        self.metrics = HypothesisMetrics()

    # --- Core API ---

    def on_target_visible(self, track_id: str) -> None:
        """Called when target is detected — purge hypotheses."""
        self._hypotheses.pop(track_id, None)
        self._merge_pairs.pop(track_id, None)
        self._behavior_hints.pop(track_id, None)

    def set_behavior_hint(self, track_id: str, evasive: bool = False,
                          concealment_seeking: bool = False) -> None:
        self._behavior_hints[track_id] = {
            "evasive": evasive,
            "concealment_seeking": concealment_seeking,
        }

    def on_target_lost(
        self,
        track_id: str,
        bbox: tuple[float, float, float, float],
        world_lat: float,
        world_lon: float,
        world_speed_mps: float,
        world_heading_rad: float,
        total_hits: int,
        smoothed_confidence: float,
        frame_w: int,
        frame_h: int,
        all_tracks: list[dict],
        drone_pose: DronePose,
        appearance_stability: float = 0.0,
    ) -> None:
        """Generate initial hypothesis set for a newly lost target."""
        hyps = self._generate_initial_hypotheses(
            track_id=track_id,
            bbox=bbox,
            world_lat=world_lat,
            world_lon=world_lon,
            world_speed_mps=world_speed_mps,
            world_heading_rad=world_heading_rad,
            total_hits=total_hits,
            smoothed_confidence=smoothed_confidence,
            frame_w=frame_w,
            frame_h=frame_h,
            all_tracks=all_tracks,
            drone_pose=drone_pose,
            appearance_stability=appearance_stability,
        )
        self._hypotheses[track_id] = hyps
        self._normalize(track_id)
        self._prune(track_id)

        # Apply behavior hints
        hints = self._behavior_hints.get(track_id, {})
        if hints.get("evasive"):
            for h in self._hypotheses.get(track_id, []):
                h.search_radius_m *= 1.5
        if hints.get("concealment_seeking"):
            for h in self._hypotheses.get(track_id, []):
                if h.htype == HypothesisType.OCCLUDED:
                    h.probability *= 1.3
            self._normalize(track_id)

    def evolve(self, track_id: str, drone_pose: DronePose, dt: float) -> None:
        """Evolve hypothesis probabilities for one frame."""
        hyps = self._hypotheses.get(track_id)
        if not hyps:
            return

        for h in hyps:
            h.age_frames += 1
            self._decay_hypothesis(h, drone_pose, track_id)
            self._extrapolate_position(h, dt)

        self._normalize(track_id)
        self._prune(track_id)
        self.metrics.record_hypothesis_count(len(hyps))

    def on_target_recovered(
        self,
        track_id: str,
        recovered_lat: float,
        recovered_lon: float,
    ) -> None:
        """Record which hypothesis was correct when target reappears."""
        hyps = self._hypotheses.get(track_id)
        if not hyps:
            return

        best = max(hyps, key=lambda h: h.probability)
        dist_m = self._haversine_m(
            best.predicted_lat, best.predicted_lon,
            recovered_lat, recovered_lon,
        )
        correct = dist_m < best.search_radius_m * 2.0
        self.metrics.record_recovery(best.htype.value, correct)
        self.on_target_visible(track_id)

    # --- Query API ---

    def get_hypotheses(self, track_id: str) -> list[TargetHypothesis]:
        return self._hypotheses.get(track_id, [])

    def get_best_hypothesis(self, track_id: str) -> TargetHypothesis | None:
        hyps = self._hypotheses.get(track_id)
        if not hyps:
            return None
        return max(hyps, key=lambda h: h.probability)

    def get_search_sectors(self, track_id: str) -> list[dict]:
        """Get probability-weighted search sectors for a target."""
        hyps = self._hypotheses.get(track_id)
        if not hyps:
            return []
        sectors = []
        for h in hyps:
            if h.age_frames > 0:
                # Bearing from last known position to predicted position
                dn, de = latlon_to_local(
                    h.predicted_lat, h.predicted_lon,
                    h.predicted_lat, h.predicted_lon,  # origin = predicted pos itself
                )
                # We need bearing FROM last known TO predicted, but we don't store origin here
                # Instead just use heading as bearing estimate
                bearing = math.degrees(h.heading_rad)
            else:
                bearing = math.degrees(h.heading_rad)
            width = self._sector_width(h)
            sectors.append({
                "bearing_deg": round(bearing, 1),
                "width_deg": width,
                "probability": round(h.probability, 3),
                "source_type": h.htype.value,
                "predicted_lat": round(h.predicted_lat, 6),
                "predicted_lon": round(h.predicted_lon, 6),
                "search_radius_m": round(h.search_radius_m, 1),
            })
        return sectors

    def get_merged_groups(self) -> list[tuple[str, str]]:
        """Return pairs of track IDs currently in a merge hypothesis."""
        pairs = []
        seen = set()
        for tid, partners in self._merge_pairs.items():
            for pid in partners:
                key = tuple(sorted([tid, pid]))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        return pairs

    def get_merge_partners(self, track_id: str) -> list[str]:
        return self._merge_pairs.get(track_id, [])

    def get_max_hypothesis_probability(self, track_id: str) -> float:
        h = self.get_best_hypothesis(track_id)
        return h.probability if h else 0.0

    # --- Internal: hypothesis generation ---

    def _generate_initial_hypotheses(
        self,
        track_id: str,
        bbox: tuple[float, float, float, float],
        world_lat: float,
        world_lon: float,
        world_speed_mps: float,
        world_heading_rad: float,
        total_hits: int,
        smoothed_confidence: float,
        frame_w: int,
        frame_h: int,
        all_tracks: list[dict],
        drone_pose: DronePose,
        appearance_stability: float,
    ) -> list[TargetHypothesis]:
        hyps = []

        # 1. OCCLUDED — vegetation-based occlusion
        occ_prob = 0.0
        if self._terrain.zone_count > 0 and world_lat != 0.0:
            occ_prob = self._terrain.occlusion_probability(
                world_lat, world_lon,
                drone_pose.alt_m, drone_pose.lat, drone_pose.lon,
            )
        occ_base = 0.3 + 0.3 * occ_prob  # 0.3 — 0.6
        hyps.append(TargetHypothesis(
            htype=HypothesisType.OCCLUDED,
            predicted_lat=world_lat,
            predicted_lon=world_lon,
            velocity_mps=0.0,
            heading_rad=world_heading_rad,
            probability=occ_base,
            occlusion_prob=occ_prob,
            appearance_consistency=appearance_stability,
            search_radius_m=10.0,
        ))

        # 2. CONTINUED_PATH — velocity extrapolation
        speed, heading = world_speed_mps, world_heading_rad
        if self._terrain.zone_count > 0:
            speed, heading = self._terrain.movement_prior(
                world_lat, world_lon, heading, speed,
            )
        movement_consistency = min(1.0, total_hits / 30.0)
        path_base = 0.3 * (0.5 + 0.5 * movement_consistency)
        # Predict position 1 second ahead
        dn = speed * 1.0 * math.cos(heading)
        de = speed * 1.0 * math.sin(heading)
        pred_lat, pred_lon = local_to_latlon(dn, de, world_lat, world_lon)
        hyps.append(TargetHypothesis(
            htype=HypothesisType.CONTINUED_PATH,
            predicted_lat=pred_lat,
            predicted_lon=pred_lon,
            velocity_mps=speed,
            heading_rad=heading,
            probability=path_base,
            appearance_consistency=appearance_stability,
            search_radius_m=max(5.0, speed * 2.0),
        ))

        # 3. EXITED_FOV — near frame edge
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        edge_dist = min(cx / frame_w, 1.0 - cx / frame_w, cy / frame_h, 1.0 - cy / frame_h)
        fov_base = 0.4 if edge_dist < 0.15 else 0.1
        hyps.append(TargetHypothesis(
            htype=HypothesisType.EXITED_FOV,
            predicted_lat=world_lat,
            predicted_lon=world_lon,
            velocity_mps=world_speed_mps,
            heading_rad=world_heading_rad,
            probability=fov_base,
            appearance_consistency=0.0,
            search_radius_m=max(10.0, world_speed_mps * 3.0),
        ))

        # 4. MERGED_GROUP — nearby same-class overlapping tracks
        merge_base = 0.0
        merge_partners = []
        for t in all_tracks:
            other_id = t.get("track_id", "")
            if other_id == track_id:
                continue
            if t.get("status") == "DELETED":
                continue
            ob = t.get("bbox", [0, 0, 1, 1])
            iou = _bbox_iou(bbox, tuple(ob))
            if iou > 0.3:
                merge_base = 0.2
                merge_partners.append(other_id)
        if merge_partners:
            self._merge_pairs[track_id] = merge_partners
            hyps.append(TargetHypothesis(
                htype=HypothesisType.MERGED_GROUP,
                predicted_lat=world_lat,
                predicted_lon=world_lon,
                velocity_mps=0.0,
                heading_rad=0.0,
                probability=merge_base,
                appearance_consistency=appearance_stability,
                search_radius_m=5.0,
            ))

        # 5. STATIONARY — target was barely moving
        stat_base = 0.0
        if world_speed_mps < 0.3 and total_hits > 10:
            stat_base = 0.4
        elif world_speed_mps < 1.0:
            stat_base = 0.1
        if stat_base > 0:
            hyps.append(TargetHypothesis(
                htype=HypothesisType.STATIONARY,
                predicted_lat=world_lat,
                predicted_lon=world_lon,
                velocity_mps=0.0,
                heading_rad=0.0,
                probability=stat_base,
                appearance_consistency=appearance_stability,
                search_radius_m=3.0,
            ))

        return hyps

    # --- Internal: evolution ---

    def _decay_hypothesis(self, h: TargetHypothesis, drone_pose: DronePose,
                          track_id: str = "") -> None:
        if h.htype == HypothesisType.OCCLUDED:
            hints = self._behavior_hints.get(track_id, {})
            if hints.get("concealment_seeking"):
                decay = 0.999
            else:
                decay = 0.998 if h.occlusion_prob > 0.3 else 0.995
            h.probability *= decay

        elif h.htype == HypothesisType.CONTINUED_PATH:
            terrain_type = "open"
            if self._terrain.zone_count > 0 and h.predicted_lat != 0.0:
                terrain_type = self._terrain.terrain_type_at(
                    h.predicted_lat, h.predicted_lon,
                )
            decay = 0.993 if terrain_type == "road" else 0.990
            h.probability *= decay

        elif h.htype == HypothesisType.EXITED_FOV:
            h.probability *= 1.01
            h.probability = min(h.probability, 0.5)

        elif h.htype == HypothesisType.MERGED_GROUP:
            decay = 0.995 if h.age_frames < 30 else 0.98
            h.probability *= decay

        elif h.htype == HypothesisType.STATIONARY:
            h.probability *= 1.002
            h.probability = min(h.probability, 0.6)

    def _extrapolate_position(self, h: TargetHypothesis, dt: float) -> None:
        if h.velocity_mps < 0.01:
            return
        # Decelerate: 0.97^age
        effective_speed = h.velocity_mps * (0.97 ** h.age_frames)
        dn = effective_speed * dt * math.cos(h.heading_rad)
        de = effective_speed * dt * math.sin(h.heading_rad)
        if abs(dn) > 0.001 or abs(de) > 0.001:
            h.predicted_lat, h.predicted_lon = local_to_latlon(
                dn, de, h.predicted_lat, h.predicted_lon,
            )
        # Expand search radius with age
        h.search_radius_m += 0.2

    # --- Internal: normalization & pruning ---

    def _normalize(self, track_id: str) -> None:
        hyps = self._hypotheses.get(track_id)
        if not hyps:
            return
        total = sum(h.probability for h in hyps)
        if total > 0:
            for h in hyps:
                h.probability /= total

    def _prune(self, track_id: str) -> None:
        hyps = self._hypotheses.get(track_id)
        if not hyps:
            return
        # Drop below threshold
        hyps = [h for h in hyps if h.probability >= MIN_PROB]
        # Cap count — keep highest probability
        if len(hyps) > MAX_HYPOTHESES:
            hyps.sort(key=lambda h: h.probability, reverse=True)
            hyps = hyps[:MAX_HYPOTHESES]
        self._hypotheses[track_id] = hyps

    # --- Internal: utilities ---

    def _sector_width(self, h: TargetHypothesis) -> float:
        if h.htype == HypothesisType.OCCLUDED:
            return 15.0 if h.probability > 0.5 else 25.0
        elif h.htype == HypothesisType.EXITED_FOV:
            return 40.0
        elif h.htype == HypothesisType.CONTINUED_PATH:
            return 20.0
        return 30.0

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
