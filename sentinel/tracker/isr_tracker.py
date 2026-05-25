"""ISR-grade persistent multi-object tracker for UAV aerial surveillance.

BoT-SORT-inspired tracker combining:
- 8-state Kalman filter for smooth motion prediction
- Camera motion compensation (CMC) for drone movement
- Spatial-color appearance features for re-identification
- 3-stage cascade matching with Hungarian algorithm
- Proper track lifecycle: TENTATIVE → CONFIRMED → LOST → DELETED

Designed for aerial top-down perspective with moving camera.
Preserves stable target IDs through occlusions and temporary disappearances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .appearance import AppearanceGallery, extract_features, extract_identity_descriptor
from .cmc import CameraMotionCompensator
from .kalman_filter import KalmanFilter
from .matching import cascade_match
from .reid import IdentityMemory, ReIDMetrics, SemanticIdentity
from .hypothesis import HypothesisManager
from .behavior import BehaviorAnalyzer
from .priority import PriorityScorer
from .terrain_reasoning import TerrainReasoner
from .world_projection import (
    CameraParams,
    DronePose,
    WorldEstimate,
    estimate_world_velocity,
    project_bbox_to_world,
)


@dataclass
class ISRTrackState:
    """Internal state for a single tracked target."""

    track_id: str
    class_name: str
    state: str = "TENTATIVE"  # TENTATIVE, CONFIRMED, LOST
    hits: int = 0
    missed_frames: int = 0
    total_hits: int = 0
    confidence: float = 0.0
    smoothed_confidence: float = 0.0

    bbox: tuple[float, float, float, float] = (0, 0, 1, 1)
    predicted_bbox: tuple[float, float, float, float] = (0, 0, 1, 1)

    velocity: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0

    first_frame: int = 0
    last_frame: int = 0
    frame_id: int = 0

    trajectory: list[tuple[float, float]] = field(default_factory=list)
    trajectory_max: int = 100

    kalman: KalmanFilter | None = None
    appearance_gallery: AppearanceGallery = field(default_factory=lambda: AppearanceGallery(max_size=10))

    recovered_frames_ago: int = 0  # >0 means recently recovered, decrements each frame
    _identity_descriptor: dict = field(default_factory=dict)
    semantic_identity: SemanticIdentity | None = None

    # World-space state (persistent across camera motion)
    world_estimate: WorldEstimate | None = None
    world_speed_mps: float = 0.0
    world_heading_rad: float = 0.0
    world_uncertainty_m: float = 0.0
    prev_world_estimate: WorldEstimate | None = None
    occlusion_state: str = "VISIBLE"  # VISIBLE, OCCLUDED, SEARCHING, REACQUIRED, STALE
    occluded_frames: int = 0
    world_trail: list[tuple[float, float]] = field(default_factory=list)
    world_trail_max: int = 200

    def push_trajectory(self, cx: float, cy: float) -> None:
        self.trajectory.append((round(cx, 1), round(cy, 1)))
        if len(self.trajectory) > self.trajectory_max:
            self.trajectory = self.trajectory[-self.trajectory_max:]


class ISRTracker:
    """Persistent ISR multi-object tracker."""

    def __init__(
        self,
        max_lost_frames: int = 150,
        min_hits_confirm: int = 3,
        high_conf_threshold: float = 0.5,
        fast_confirm_confidence: float = 0.7,
        iou_gate: float = 0.8,
        lost_iou_gate: float = 0.9,
        iou_weight: float = 0.7,
        appearance_weight: float = 0.3,
        cmc_enabled: bool = True,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 0.1,
        trajectory_length: int = 100,
        bbox_ema_alpha: float = 0.3,
        conf_ema_alpha: float = 0.3,
        fps: float = 15.0,
        camera: CameraParams | None = None,
    ) -> None:
        self._max_lost = max_lost_frames
        self._min_hits = min_hits_confirm
        self._high_conf = high_conf_threshold
        self._fast_confirm = fast_confirm_confidence
        self._iou_gate = iou_gate
        self._lost_iou_gate = lost_iou_gate
        self._iou_weight = iou_weight
        self._app_weight = appearance_weight
        self._kalman_q = kalman_process_noise
        self._kalman_r = kalman_measurement_noise
        self._traj_len = trajectory_length
        self._bbox_alpha = bbox_ema_alpha
        self._conf_alpha = conf_ema_alpha
        self._fps = fps

        self._tracks: list[ISRTrackState] = []
        self._next_id = 1
        self._cmc = CameraMotionCompensator(enabled=cmc_enabled)
        self._total_recoveries = 0
        self._total_id_switches = 0
        self._identity_memory = IdentityMemory(max_entries=50, timeout_seconds=30.0)
        self._reid_metrics = ReIDMetrics()
        self._reid_threshold = 0.6

        self._camera = camera or CameraParams()
        self._drone_pose = DronePose()
        self.terrain = TerrainReasoner()
        self._hypothesis_mgr = HypothesisManager(self.terrain, fps)
        self._behavior = BehaviorAnalyzer(self.terrain, fps)
        self._priority_scorer = PriorityScorer(self._behavior)

        self._projection_successes: int = 0
        self._projection_failures: int = 0
        self._projection_fail_reasons: dict[str, int] = {}

    def _make_id(self) -> str:
        tid = f"TGT-{self._next_id:03d}"
        self._next_id += 1
        return tid

    def _bbox_center(self, bbox: tuple) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def update_drone_pose(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        heading_deg: float,
        groundspeed_mps: float = 0.0,
    ) -> None:
        """Update drone telemetry for world-space projection."""
        self._drone_pose = DronePose(
            lat=lat, lon=lon, alt_m=alt_m,
            heading_deg=heading_deg,
            groundspeed_mps=groundspeed_mps,
        )

    def update_camera_params(self, camera: CameraParams) -> None:
        """Update camera intrinsics at runtime (e.g., after discovering actual frame dimensions)."""
        self._camera = camera

    def _create_track(
        self,
        bbox: tuple[float, float, float, float],
        class_name: str,
        confidence: float,
        frame_id: int,
        feature: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> ISRTrackState:
        cx, cy = self._bbox_center(bbox)
        kf = KalmanFilter(bbox, self._kalman_q, self._kalman_r)

        track = ISRTrackState(
            track_id=self._make_id(),
            class_name=class_name,
            confidence=confidence,
            smoothed_confidence=confidence,
            bbox=bbox,
            predicted_bbox=bbox,
            first_frame=frame_id,
            last_frame=frame_id,
            frame_id=frame_id,
            kalman=kf,
            trajectory_max=self._traj_len,
        )
        track.push_trajectory(cx, cy)
        if feature is not None:
            track.appearance_gallery.update(feature)

        # Initialize semantic identity
        track.semantic_identity = SemanticIdentity(
            track_id=track.track_id, class_name=class_name,
        )
        if feature is not None and frame is not None:
            bbox_int = tuple(int(v) for v in bbox)
            track.semantic_identity.update_appearance(feature, frame, bbox_int, frame_id)
        self._reid_metrics.record_identity_start(track.track_id, frame_id)

        return track

    def update(
        self,
        detections: list[dict],
        frame: np.ndarray | None = None,
        frame_id: int = 0,
    ) -> list[dict]:
        """Process detections and return current track states.

        Args:
            detections: List of dicts with 'bbox', 'class', 'confidence'
            frame: Current BGR frame (for CMC and appearance features)
            frame_id: Sequential frame number

        Returns:
            List of track state dicts for overlay rendering.
        """
        # Step 0: CMC update
        if frame is not None:
            self._cmc.update(frame)

        # Step 0.5: Extract appearance features
        for d in detections:
            bbox_int = tuple(int(v) for v in d["bbox"])
            d["_feature"] = extract_features(frame, bbox_int) if frame is not None else None
            d["_bbox"] = tuple(float(v) for v in d["bbox"])

        # Step 1: Predict all tracks forward + apply CMC
        for track in self._tracks:
            if track.kalman is not None:
                track.predicted_bbox = track.kalman.predict(dt=1.0)
                if self._cmc.enabled:
                    track.kalman.apply_warp(self._cmc.warp_matrix)
                    track.predicted_bbox = track.kalman.get_state_bbox()

        # Step 2: Cascade matching
        fw = frame.shape[1] if frame is not None else 1280
        fh = frame.shape[0] if frame is not None else 720
        matched, unmatched_det_idx, unmatched_track_idx = cascade_match(
            detections=detections,
            tracks=self._tracks,
            high_conf_threshold=self._high_conf,
            iou_gate=self._iou_gate,
            lost_iou_gate=self._lost_iou_gate,
            iou_weight=self._iou_weight,
            appearance_weight=self._app_weight,
            identity_memory=self._identity_memory,
            reid_threshold=self._reid_threshold,
            frame_w=fw,
            frame_h=fh,
        )

        # Step 3: Update matched tracks
        matched_det_set = set()
        matched_track_set = set()
        for di, ti in matched:
            det = detections[di]
            track = self._tracks[ti]
            matched_det_set.add(di)
            matched_track_set.add(ti)

            was_lost = track.state == "LOST"
            new_bbox = det["_bbox"]
            old_cx, old_cy = self._bbox_center(track.bbox)
            new_cx, new_cy = self._bbox_center(new_bbox)

            # Kalman update
            if track.kalman is not None:
                track.kalman.update(new_bbox)
                track.bbox = track.kalman.get_state_bbox()
                track.velocity = track.kalman.get_velocity()
                track.speed = track.kalman.get_speed()
            else:
                track.bbox = new_bbox
                track.velocity = (new_cx - old_cx, new_cy - old_cy)
                track.speed = math.sqrt(track.velocity[0] ** 2 + track.velocity[1] ** 2)

            # Heading from velocity
            if track.speed > 0.5:
                track.heading = math.atan2(track.velocity[1], track.velocity[0])

            track.confidence = det["confidence"]
            track.smoothed_confidence = (
                self._conf_alpha * det["confidence"]
                + (1 - self._conf_alpha) * track.smoothed_confidence
            )
            track.hits += 1
            track.total_hits += 1
            track.missed_frames = 0
            track.last_frame = frame_id
            track.frame_id = frame_id
            track.predicted_bbox = track.bbox

            # Appearance update
            if det.get("_feature") is not None:
                track.appearance_gallery.update(det["_feature"])

            # Identity descriptor update (every 10 frames)
            if frame is not None and (track.total_hits % 10 == 0 or not track._identity_descriptor):
                bbox_int = tuple(int(v) for v in new_bbox)
                track._identity_descriptor = extract_identity_descriptor(frame, bbox_int)

            # Semantic identity update (spatial + appearance)
            if track.semantic_identity is not None:
                fw = frame.shape[1] if frame is not None else 1280
                fh = frame.shape[0] if frame is not None else 720
                track.semantic_identity.update_spatial(
                    new_cx, new_cy, track.velocity, track.heading,
                    track.speed, fw, fh, frame_id,
                )
                if det.get("_feature") is not None and frame is not None:
                    track.semantic_identity.update_appearance(
                        det["_feature"], frame,
                        tuple(int(v) for v in new_bbox), frame_id,
                    )

            # Record Re-ID recovery
            if was_lost and track.semantic_identity is not None:
                self._reid_metrics.record_reacquisition(
                    track.track_id, frame_id,
                    track.missed_frames, 0.0,
                )

            # Hypothesis: record recovery or mark visible
            if was_lost:
                if track.world_estimate and track.world_estimate.valid:
                    self._hypothesis_mgr.on_target_recovered(
                        track.track_id,
                        track.world_estimate.lat,
                        track.world_estimate.lon,
                    )
                else:
                    self._hypothesis_mgr.on_target_visible(track.track_id)
            else:
                self._hypothesis_mgr.on_target_visible(track.track_id)

            # State transitions
            if track.state == "TENTATIVE":
                if track.hits >= self._min_hits:
                    track.state = "CONFIRMED"
                elif det["confidence"] >= self._fast_confirm and track.hits >= 1:
                    track.state = "CONFIRMED"
            elif track.state == "LOST":
                track.state = "CONFIRMED"
                track.recovered_frames_ago = 10
                self._total_recoveries += 1
            elif was_lost:
                track.recovered_frames_ago = 10

            track.push_trajectory(
                (track.bbox[0] + track.bbox[2]) / 2.0,
                (track.bbox[1] + track.bbox[3]) / 2.0,
            )

            # World-space projection
            if not math.isnan(self._drone_pose.lon) and math.isfinite(self._drone_pose.lat):
                track.prev_world_estimate = track.world_estimate
                track.world_estimate = project_bbox_to_world(
                    new_bbox, self._drone_pose, self._camera,
                )
                if track.world_estimate.valid:
                    self._projection_successes += 1
                else:
                    self._projection_failures += 1
                    reason = track.world_estimate.failure_reason or "UNKNOWN"
                    self._projection_fail_reasons[reason] = self._projection_fail_reasons.get(reason, 0) + 1
                if track.world_estimate.valid and track.prev_world_estimate is not None and track.prev_world_estimate.valid:
                    track.world_speed_mps, track.world_heading_rad, track.world_uncertainty_m = (
                        estimate_world_velocity(
                            track.prev_world_estimate, track.world_estimate,
                            1.0 / self._fps,
                        )
                    )
                if track.world_estimate.valid:
                    track.world_trail.append(
                        (track.world_estimate.lat, track.world_estimate.lon),
                    )
                    if len(track.world_trail) > track.world_trail_max:
                        track.world_trail = track.world_trail[-track.world_trail_max:]
                    # Terrain-aware: if target is in vegetation zone, mark partial occlusion
                    if self.terrain.zone_count > 0:
                        terrain_type = self.terrain.terrain_type_at(
                            track.world_estimate.lat, track.world_estimate.lon,
                        )
                        if terrain_type == "vegetation":
                            track.occlusion_state = "REACQUIRED" if track.occlusion_state == "OCCLUDED" else "VISIBLE"
                        else:
                            track.occlusion_state = "VISIBLE"
                    else:
                        track.occlusion_state = "VISIBLE"
                track.occluded_frames = 0

        # Step 4: Age unmatched tracks
        for ti in unmatched_track_idx:
            track = self._tracks[ti]
            track.missed_frames += 1

            if track.state == "CONFIRMED" and track.missed_frames > 1:
                track.state = "LOST"
            elif track.state == "TENTATIVE" and track.missed_frames > 2:
                track.state = "DELETED"

            if track.missed_frames > self._max_lost:
                track.state = "DELETED"

            # Occlusion state machine
            if track.occlusion_state == "VISIBLE":
                track.occlusion_state = "OCCLUDED"
                track.occluded_frames = 1
            elif track.occlusion_state in ("OCCLUDED", "SEARCHING", "REACQUIRED"):
                track.occluded_frames += 1
                if track.occluded_frames > 15:
                    track.occlusion_state = "SEARCHING"
                if track.occluded_frames > self._max_lost * 0.8:
                    track.occlusion_state = "STALE"

            # Decrement recovery indicator
            if track.recovered_frames_ago > 0:
                track.recovered_frames_ago -= 1

        # Step 4.5: Hypothesis management for unmatched tracks
        for ti in unmatched_track_idx:
            track = self._tracks[ti]
            if track.state == "DELETED" or track.class_name == "":
                continue
            # Get world position for hypothesis generation
            w_lat = track.world_estimate.lat if track.world_estimate and track.world_estimate.valid else 0.0
            w_lon = track.world_estimate.lon if track.world_estimate and track.world_estimate.valid else 0.0
            if track.occluded_frames == 1:
                # Wire behavior hints before generating hypotheses
                bp = self._behavior.get_profile(track.track_id)
                if bp:
                    self._hypothesis_mgr.set_behavior_hint(
                        track.track_id,
                        evasive=bp.current_pattern.value == "EVASIVE",
                        concealment_seeking=bp.current_pattern.value == "CONCEALMENT_SEEKING",
                    )
                # Just lost — generate hypotheses
                track_dicts = [
                    {
                        "track_id": t.track_id,
                        "bbox": t.bbox,
                        "status": t.state,
                        "class": t.class_name,
                    }
                    for t in self._tracks
                ]
                self._hypothesis_mgr.on_target_lost(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    world_lat=w_lat,
                    world_lon=w_lon,
                    world_speed_mps=track.world_speed_mps,
                    world_heading_rad=track.world_heading_rad,
                    total_hits=track.total_hits,
                    smoothed_confidence=track.smoothed_confidence,
                    frame_w=fw,
                    frame_h=fh,
                    all_tracks=track_dicts,
                    drone_pose=self._drone_pose,
                    appearance_stability=(
                        track.semantic_identity.appearance_stability
                        if track.semantic_identity else 0.0
                    ),
                )
            else:
                self._hypothesis_mgr.evolve(
                    track.track_id, self._drone_pose, 1.0 / self._fps,
                )

        # Step 5: Create new tracks for truly unmatched detections
        for di in unmatched_det_idx:
            det = detections[di]

            # Cross-lifecycle Re-ID: check identity memory before creating new ID
            reid_original_id = det.get("_reid_match", {}).get("original_track_id")
            if reid_original_id is None and frame is not None and det.get("_feature") is not None:
                cx = (det["_bbox"][0] + det["_bbox"][2]) / 2.0
                cy = (det["_bbox"][1] + det["_bbox"][3]) / 2.0
                fw = frame.shape[1]
                fh = frame.shape[0]
                matched_id, score, breakdown = self._identity_memory.find_match(
                    det["_feature"], cx, cy, 0.0, fw, fh,
                    det["class"], self._reid_threshold,
                )
                if matched_id is not None:
                    reid_original_id = matched_id
                    self._reid_metrics.record_cross_lifecycle_match(
                        matched_id, "?", frame_id, score,
                    )

            new_track = self._create_track(
                det["_bbox"], det["class"], det["confidence"],
                frame_id, det.get("_feature"), frame,
            )

            # Restore archived track ID from cross-lifecycle match
            if reid_original_id is not None:
                new_track.track_id = reid_original_id
                if new_track.semantic_identity is not None:
                    new_track.semantic_identity.track_id = reid_original_id

            self._tracks.append(new_track)

        # Step 6: Purge DELETED tracks (archive identities first)
        for t in self._tracks:
            if t.state == "DELETED":
                if t.semantic_identity is not None and t.total_hits >= 3:
                    self._identity_memory.archive(t.semantic_identity)
                    self._reid_metrics.record_identity_end(t.track_id, frame_id)
                self._behavior.remove_profile(t.track_id)
        self._tracks = [t for t in self._tracks if t.state != "DELETED"]

        # Step 7: Build output
        result = self._build_output(frame_id)

        # Step 7.5: Behavioral analysis
        self._behavior.update(result)
        return result

    def _emergence_points_for_track(self, t: ISRTrackState) -> list[dict]:
        """Compute emergence points for an occluded track."""
        if t.occlusion_state not in ("OCCLUDED", "SEARCHING"):
            return []
        if t.world_estimate is None or not t.world_estimate.valid:
            return []
        points = self.terrain.predict_emergence_points(
            t.world_estimate.lat, t.world_estimate.lon,
            t.world_speed_mps, t.world_heading_rad,
            t.occluded_frames, self._fps,
        )
        return [
            {"lat": round(p.lat, 6), "lon": round(p.lon, 6),
             "probability": round(p.probability, 3), "source": p.source,
             "radius_m": round(p.radius_m, 1)}
            for p in points
        ]

    def _build_output(self, frame_id: int) -> list[dict]:
        """Convert track states to overlay-friendly dicts."""
        result = []
        for t in self._tracks:
            cx = (t.bbox[0] + t.bbox[2]) / 2.0
            cy = (t.bbox[1] + t.bbox[3]) / 2.0
            age = frame_id - t.first_frame

            # Skip tentative tracks from overlay unless they have high confidence
            if t.state == "TENTATIVE" and t.smoothed_confidence < self._fast_confirm:
                continue

            # Identity descriptor from latest appearance
            identity_desc = ""
            if t._identity_descriptor:
                identity_desc = (
                    f"{t._identity_descriptor.get('upper_color', '?')}/"
                    f"{t._identity_descriptor.get('lower_color', '?')}"
                )

            # Semantic identity data
            identity_stability = 0.0
            predicted_pos = None
            if t.semantic_identity is not None:
                identity_stability = t.semantic_identity.appearance_stability
                if t.state == "LOST":
                    pred = t.semantic_identity.predict_position(t.missed_frames)
                    if pred is not None:
                        predicted_pos = [round(v, 1) for v in pred]

            # World-space data
            world_pos = None
            world_vel = None
            if t.world_estimate is not None and t.world_estimate.valid:
                world_pos = {
                    "lat": round(t.world_estimate.lat, 6),
                    "lon": round(t.world_estimate.lon, 6),
                    "north_m": round(t.world_estimate.north_m, 1),
                    "east_m": round(t.world_estimate.east_m, 1),
                    "ground_distance_m": round(t.world_estimate.ground_distance_m, 1),
                    "uncertainty_m": round(t.world_estimate.uncertainty_m, 1),
                }
                world_vel = {
                    "speed_mps": round(t.world_speed_mps, 2),
                    "heading_rad": round(t.world_heading_rad, 3),
                    "heading_deg": round(math.degrees(t.world_heading_rad), 1),
                }

            result.append({
                "class": t.class_name,
                "confidence": round(t.smoothed_confidence, 3),
                "bbox": [round(v, 1) for v in t.bbox],
                "track_id": t.track_id,
                "status": t.state,
                "age_frames": age,
                "hits": t.total_hits,
                "missed_frames": t.missed_frames,
                "velocity": [round(t.velocity[0], 2), round(t.velocity[1], 2)],
                "speed": round(t.speed, 2),
                "heading": round(t.heading, 3),
                "trajectory": t.trajectory[-50:],
                "recovered": t.recovered_frames_ago > 0,
                "recovered_frames_ago": t.recovered_frames_ago,
                "identity_descriptor": identity_desc,
                "identity_detail": t._identity_descriptor,
                "identity_stability": round(identity_stability, 3),
                "predicted_position": predicted_pos,
                "occlusion_state": t.occlusion_state,
                "occluded_frames": t.occluded_frames,
                "world_position": world_pos,
                "world_velocity": world_vel,
                "world_uncertainty_m": round(t.world_uncertainty_m, 1),
                "projection_status": t.world_estimate.failure_reason if t.world_estimate else "NEVER_ATTEMPTED",
                "world_trail": [(round(la, 6), round(lo, 6)) for la, lo in t.world_trail[-100:]],
                "emergence_points": self._emergence_points_for_track(t),
                "hypotheses": [
                    {"type": h.htype.value, "probability": round(h.probability, 3),
                     "predicted_lat": round(h.predicted_lat, 6),
                     "predicted_lon": round(h.predicted_lon, 6),
                     "search_radius_m": round(h.search_radius_m, 1),
                     "age_frames": h.age_frames}
                    for h in self._hypothesis_mgr.get_hypotheses(t.track_id)
                ],
                "best_hypothesis": (bh.htype.value if (bh := self._hypothesis_mgr.get_best_hypothesis(t.track_id)) else None),
                "search_sectors": self._hypothesis_mgr.get_search_sectors(t.track_id),
                "merge_group": self._hypothesis_mgr.get_merge_partners(t.track_id),
            })
            # Behavioral data (from previous frame's analysis)
            bp = self._behavior.get_profile(t.track_id)
            result[-1]["behavior_pattern"] = bp.current_pattern.value if bp else ""
            result[-1]["concealment_affinity"] = round(bp.concealment_affinity, 3) if bp else 0.0
            result[-1]["anomaly_score"] = round(bp.anomaly_score, 3) if bp else 0.0
            result[-1]["interaction_count"] = bp.interaction_count if bp else 0
            result[-1]["priority_score"] = round(self._priority_scorer.score_target(result[-1]), 3)
        return result

    def get_track(self, track_id: str) -> ISRTrackState | None:
        """Get a specific track by ID."""
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None

    def get_metrics(self) -> dict:
        """Return tracker metrics."""
        active = sum(1 for t in self._tracks if t.state == "CONFIRMED")
        lost = sum(1 for t in self._tracks if t.state == "LOST")
        tentative = sum(1 for t in self._tracks if t.state == "TENTATIVE")
        with_world = sum(1 for t in self._tracks if t.world_estimate is not None and t.world_estimate.valid)
        occluded = sum(1 for t in self._tracks if t.occlusion_state in ("OCCLUDED", "SEARCHING"))
        return {
            "active": active,
            "lost": lost,
            "tentative": tentative,
            "total_tracks_created": self._next_id - 1,
            "total_recoveries": self._total_recoveries,
            "alive_tracks": len(self._tracks),
            "reid": self._reid_metrics.get_summary(),
            "identity_memory_size": self._identity_memory.size,
            "world_tracking": {
                "tracks_with_world_pos": with_world,
                "occluded_tracks": occluded,
                "drone_alt_m": self._drone_pose.alt_m,
                "drone_heading_deg": self._drone_pose.heading_deg,
                "projection_successes": self._projection_successes,
                "projection_failures": self._projection_failures,
                "projection_fail_reasons": dict(self._projection_fail_reasons),
            },
            "hyp": self._hypothesis_mgr.metrics.get_summary(),
            "behavior": self._behavior.metrics.get_summary(),
        }

    @property
    def behavior_analyzer(self) -> BehaviorAnalyzer:
        return self._behavior

    def emergence_points_for(self, track_id: str) -> list[dict]:
        """Get predicted emergence points for an occluded track."""
        track = self.get_track(track_id)
        if track is None or track.world_estimate is None or not track.world_estimate.valid:
            return []
        if track.occlusion_state not in ("OCCLUDED", "SEARCHING"):
            return []
        points = self.terrain.predict_emergence_points(
            track.world_estimate.lat,
            track.world_estimate.lon,
            track.world_speed_mps,
            track.world_heading_rad,
            track.occluded_frames,
            self._fps,
        )
        return [
            {"lat": round(p.lat, 6), "lon": round(p.lon, 6),
             "probability": round(p.probability, 3), "source": p.source,
             "radius_m": round(p.radius_m, 1)}
            for p in points
        ]
