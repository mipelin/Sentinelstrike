"""Autonomous yaw + XY follow controller for ISR UAV target tracking.

Modes:
  yaw       — Rotate to keep target horizontally centered.
  yaw_xy    — Yaw + translate to maintain target size/position.
  intercept — Approach to standoff distance, then hover/orbit.
  standoff  — SAFE observation mode: approach slowly, maintain standoff
              distance, keep altitude above target, orbit on arrival.
              Never collides. Stops forward motion when target lost.

State machine:
  UNLOCKED → LOCKED → SEARCHING → REACQUIRED → LOCKED
                      → LOST → UNLOCKED
                      → CANDIDATE → LOCKED

Standoff sub-states (in intercept_state field):
  APPROACHING — far from standoff, creeping closer
  SAFE        — at standoff distance, holding position
  ORBITING    — at standoff, orbiting at standoff yaw rate
  HOLD        — target lost, holding position + yaw search only

Thread safety: follow_loop() runs in a background thread. All state mutations
go through _lock. The tracker calls update_target() from the main thread.
"""

from __future__ import annotations

import math
import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from .pid import PIDController

if TYPE_CHECKING:
    from sentinel.flight.telemetry_cache import TelemetryCache
    from sentinel.runtime.blackboard import LatestSlot
    from sentinel.runtime.contracts import VelocityCommand


class FollowMode(str, Enum):
    YAW = "yaw"
    YAW_XY = "yaw_xy"
    INTERCEPT = "intercept"
    STANDOFF = "standoff"
    MANUAL_AIM = "manual_aim"


class TargetState(str, Enum):
    UNLOCKED = "UNLOCKED"
    LOCKED = "LOCKED"
    SUSPECT = "SUSPECT"
    SEARCHING = "SEARCHING"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"
    CANDIDATE = "CANDIDATE"


@dataclass
class FollowDiagnostics:
    """Snapshot of follow controller state for overlay rendering."""
    state: TargetState = TargetState.UNLOCKED
    follow_mode: FollowMode = FollowMode.YAW
    target_id: str = ""
    identity_descriptor: str = ""
    identity_similarity: float = 0.0
    error_x: float = 0.0
    error_y: float = 0.0
    yaw_rate: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    pid_state: dict = None
    offboard_active: bool = False
    frames_since_update: int = 0
    target_bbox: list | None = None
    target_speed: float = 0.0
    target_confidence: float = 0.0
    search_direction: float = 0.0
    target_lost_s: float = 0.0
    bbox_height_px: float = 0.0
    size_error: float = 0.0
    geofence_ok: bool = True
    distance_from_home_m: float = 0.0
    standoff_distance_ok: bool = False
    standoff_phase: str = ""
    standoff_target_m: float = 0.0
    standoff_altitude_m: float = 0.0
    standoff_actual_m: float = 0.0
    deterrence_active: bool = False
    world_position: dict | None = None
    world_velocity: dict | None = None
    occlusion_state: str = ""
    emergence_points: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    best_hypothesis: str = ""
    merge_group_partners: list[str] = field(default_factory=list)
    search_sectors: list[dict] = field(default_factory=list)
    priority_score: float = 0.0
    behavior_pattern: str = ""


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class FollowController:
    """Manages autonomous target following with multiple safety modes."""

    def __init__(
        self,
        backend=None,
        *,
        command_slot: LatestSlot[VelocityCommand] | None = None,
        telemetry_cache: TelemetryCache | None = None,
        follow_mode: FollowMode = FollowMode.YAW,
        kp: float = 0.008,
        ki: float = 0.0001,
        kd: float = 0.003,
        deadband_px: float = 20.0,
        max_yaw_rate: float = 0.6,
        lost_timeout_s: float = 8.0,
        search_yaw_rate: float = 0.25,
        decay_rate: float = 0.85,
        offboard_hz: float = 50.0,
        min_yaw_rate: float = 0.05,
        search_coast_s: float = 3.0,
        # XY follow params
        target_height_px: float = 120.0,
        max_vxy: float = 0.8,
        fwd_kp: float = 0.004,
        lat_kp: float = 0.003,
        yaw_align_threshold_px: float = 80.0,
        min_confidence: float = 0.3,
        geofence_radius_m: float = 50.0,
        # Identity-aware search
        identity_threshold: float = 0.65,
        candidate_threshold: float = 0.45,
        search_predict_s: float = 2.0,
        # Intercept / standoff params
        standoff_bbox_height_px: float = 200.0,
        orbit_yaw_rate: float = 0.15,
        # Standoff-specific params
        standoff_distance_m: float = 10.0,
        standoff_altitude_m: float = 10.0,
        standoff_approach_speed: float = 0.5,
        standoff_hold_zone: float = 0.15,
        standoff_deadband_m: float = 2.0,
        standoff_metric_kp: float = 0.2,
        orbit_on_arrival: bool = True,
        deterrence_marker: bool = False,
    ) -> None:
        self._backend = backend
        self._command_slot = command_slot
        self._telemetry_cache = telemetry_cache
        self._follow_mode = follow_mode
        self._pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            deadband=deadband_px,
            output_min=-max_yaw_rate,
            output_max=max_yaw_rate,
        )
        self._max_yaw_rate = max_yaw_rate
        self._lost_timeout_s = lost_timeout_s
        self._suspect_timeout_s = 0.5
        self._search_timeout_s = 1.5
        self._search_yaw_rate = search_yaw_rate
        self._decay_rate = decay_rate
        self._offboard_hz = offboard_hz
        self._min_yaw_rate = min_yaw_rate
        self._search_coast_s = search_coast_s

        self._target_height_px = target_height_px
        self._max_vxy = max_vxy
        self._fwd_kp = fwd_kp
        self._lat_kp = lat_kp
        self._yaw_align_threshold_px = yaw_align_threshold_px
        self._min_confidence = min_confidence
        self._geofence_radius_m = geofence_radius_m

        self._identity_threshold = identity_threshold
        self._candidate_threshold = candidate_threshold
        self._search_predict_s = search_predict_s

        self._standoff_bbox_h = standoff_bbox_height_px
        self._orbit_yaw_rate = orbit_yaw_rate

        # Standoff-specific
        self._standoff_distance_m = standoff_distance_m
        self._standoff_altitude_m = standoff_altitude_m
        self._standoff_approach_speed = standoff_approach_speed
        self._standoff_hold_zone = standoff_hold_zone
        self._standoff_deadband_m = standoff_deadband_m
        self._standoff_metric_kp = standoff_metric_kp
        self._orbit_on_arrival = orbit_on_arrival
        self._deterrence_marker = deterrence_marker

        self._lock = threading.Lock()
        self._state = TargetState.UNLOCKED
        self._target_id: str | None = None
        self._frame_w: int = 1280
        self._frame_h: int = 720

        self._target_cx: float | None = None
        self._target_cy: float | None = None
        self._target_bbox: list | None = None
        self._target_speed: float = 0.0
        self._target_confidence: float = 0.0
        self._tracker_time: float = 0.0
        self._target_velocity: tuple[float, float] = (0.0, 0.0)
        self._target_identity_desc: str = ""
        self._identity_similarity: float = 0.0

        self._current_yaw_rate: float = 0.0
        self._current_vx: float = 0.0
        self._current_vy: float = 0.0
        self._current_vz: float = 0.0
        self._last_error_x: float = 0.0
        self._last_size_error: float = 0.0
        self._last_bbox_height: float = 0.0
        self._search_direction: float = 1.0
        self._lost_time: float = 0.0

        self._last_known_vx: float = 0.0
        self._last_known_vy: float = 0.0
        self._last_locked_cx: float | None = None
        self._last_locked_cy: float | None = None

        self._target_identity_memory: str = ""

        # HSV vector memory for improved identity matching
        self._identity_memory_upper_hsv: list[float] = []
        self._identity_memory_lower_hsv: list[float] = []
        self._last_known_position: tuple[float, float] | None = None
        self._last_known_heading: float = 0.0

        # World-space tracking
        self._world_lat: float | None = None
        self._world_lon: float | None = None
        self._world_speed_mps: float = 0.0
        self._world_heading_rad: float = 0.0
        self._emergence_points: list[dict] = []

        # Hypothesis state
        self._hypotheses: list[dict] = []
        self._search_sectors: list[dict] = []
        self._merge_partners: list[str] = []

        # Behavioral state
        self._target_priority_score: float = 0.0
        self._target_behavior_pattern: str = ""

        # Standoff / intercept phase
        self._standoff_phase: str = ""

        # Geofence
        self._home_lat: float | None = None
        self._home_lon: float | None = None
        self._distance_from_home: float = 0.0
        self._geofence_ok: bool = True

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._offboard_engaged: bool = False
        self._manual_vx: float = 0.0
        self._manual_vy: float = 0.0
        self._manual_vz: float = 0.0

    @property
    def follow_mode(self) -> FollowMode:
        with self._lock:
            return self._follow_mode

    def set_follow_mode(self, follow_mode: FollowMode) -> None:
        with self._lock:
            self._follow_mode = follow_mode
        logger.info("Follow: mode set to {}", follow_mode.value)

    def _write_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float, source: str) -> None:
        if self._command_slot is None:
            return
        from sentinel.runtime.contracts import VelocityCommand

        self._command_slot.write(VelocityCommand(
            vx=vx,
            vy=vy,
            vz=vz,
            yawspeed=yaw_rate,
            source=source,
        ))

    def _emit_zero_command(self, source: str) -> None:
        self._write_velocity_command(0.0, 0.0, 0.0, 0.0, source)

    def set_manual_velocity(self, vx: float, vy: float, vz: float) -> None:
        with self._lock:
            self._manual_vx = vx
            self._manual_vy = vy
            self._manual_vz = vz

    @property
    def state(self) -> TargetState:
        with self._lock:
            return self._state

    @property
    def target_id(self) -> str | None:
        with self._lock:
            return self._target_id

    def lock_target(self, track_id: str) -> None:
        with self._lock:
            self._target_id = track_id
            self._state = TargetState.SEARCHING
            self._pid.reset()
            self._current_yaw_rate = 0.0
            self._current_vx = 0.0
            self._current_vy = 0.0
            self._current_vz = 0.0
            self._target_identity_memory = ""
            self._standoff_phase = ""
            self._identity_memory_upper_hsv = []
            self._identity_memory_lower_hsv = []
            self._last_known_position = None
            self._last_known_heading = 0.0
            self._lost_time = 0.0
            self._target_cx = None
            self._target_cy = None
            self._target_bbox = None
            self._target_speed = 0.0
            self._target_confidence = 0.0
            self._tracker_time = 0.0
            self._last_bbox_height = 0.0
            self._last_size_error = 0.0
            self._last_error_x = 0.0
            self._hypotheses = []
            self._search_sectors = []
            self._merge_partners = []
            self._emergence_points = []
            self._target_priority_score = 0.0
            self._target_behavior_pattern = ""
            self._world_lat = None
            self._world_lon = None
            self._world_speed_mps = 0.0
            self._world_heading_rad = 0.0
            self._target_identity_desc = ""
            self._identity_similarity = 0.0
        self._record_home_position()
        logger.info("Follow: locked on {} (mode={})", track_id, self._follow_mode.value)
        self._start_loop()

    def unlock(self) -> None:
        with self._lock:
            self._target_id = None
            self._state = TargetState.UNLOCKED
            self._target_cx = None
            self._current_yaw_rate = 0.0
            self._current_vx = 0.0
            self._current_vy = 0.0
            self._current_vz = 0.0
            self._target_identity_memory = ""
            self._standoff_phase = ""
            self._world_lat = None
            self._world_lon = None
            self._world_speed_mps = 0.0
            self._world_heading_rad = 0.0
            self._emergence_points = []
            self._manual_vx = 0.0
            self._manual_vy = 0.0
            self._manual_vz = 0.0
        self._stop_loop()
        self._emit_zero_command("follow_unlock")
        logger.info("Follow: unlocked")

    def update_target(self, tracks: list[dict], frame_w: int, frame_h: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._frame_w = frame_w
            self._frame_h = frame_h
            self._tracker_time = now

            if self._state == TargetState.UNLOCKED or self._target_id is None:
                return

            target = None
            for t in tracks:
                if t.get("track_id") == self._target_id:
                    target = t
                    break

            # Identity-based ReID: try to find target even if track_id changed.
            # Runs in any active state (not just SEARCHING/LOST).
            identity_sim = 0.0
            if target is None and self._target_identity_memory:
                best_match = None
                best_sim = 0.0
                for t in tracks:
                    # Skip tracks that already match another followed target
                    if t.get("track_id") == self._target_id:
                        continue
                    desc = t.get("identity_descriptor", "")
                    if desc and desc == self._target_identity_memory:
                        sim = 0.9
                    else:
                        detail = t.get("identity_detail", {})
                        sim = self._compute_identity_similarity(detail) if detail else 0.0
                    if sim > best_sim:
                        best_sim = sim
                        best_match = t

                if best_match and best_sim > self._candidate_threshold:
                    if best_sim > self._identity_threshold:
                        old_id = self._target_id
                        new_id = best_match.get("track_id", self._target_id)
                        target = best_match
                        identity_sim = best_sim
                        self._target_id = new_id
                        self._lost_time = 0.0
                        self._state = TargetState.REACQUIRED
                        logger.info("Follow: ReID transfer {} → {} (sim={:.2f})",
                                    old_id, new_id, best_sim)
                    elif self._state in (TargetState.SUSPECT, TargetState.SEARCHING, TargetState.LOST):
                        identity_sim = best_sim
                        self._identity_similarity = best_sim
                        self._state = TargetState.CANDIDATE

            if target is not None:
                bbox = target["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                bbox_h = bbox[3] - bbox[1]
                self._target_cx = cx
                self._target_cy = cy
                self._target_bbox = bbox
                self._target_speed = target.get("speed", 0.0)
                self._target_confidence = target.get("confidence", 0.0)
                self._last_bbox_height = bbox_h
                self._target_velocity = target.get("velocity", [0, 0])
                self._target_identity_desc = target.get("identity_descriptor", "")

                if self._target_identity_desc:
                    self._target_identity_memory = self._target_identity_desc

                # Capture HSV values for vector-based matching
                id_detail = target.get("identity_detail", {})
                if id_detail.get("upper_hsv"):
                    self._identity_memory_upper_hsv = id_detail["upper_hsv"]
                if id_detail.get("lower_hsv"):
                    self._identity_memory_lower_hsv = id_detail["lower_hsv"]

                # Save position for predictive search
                self._last_known_position = (cx, cy)
                if self._target_speed > 0.5:
                    self._last_known_heading = math.atan2(
                        self._target_velocity[1], self._target_velocity[0],
                    )

                self._last_known_vx = self._target_velocity[0]
                self._last_known_vy = self._target_velocity[1]
                self._last_locked_cx = cx
                self._last_locked_cy = cy

                # World-space data
                wp = target.get("world_position")
                if wp:
                    self._world_lat = wp.get("lat")
                    self._world_lon = wp.get("lon")
                wv = target.get("world_velocity")
                if wv:
                    self._world_speed_mps = wv.get("speed_mps", 0.0)
                    self._world_heading_rad = wv.get("heading_rad", 0.0)
                self._emergence_points = target.get("emergence_points", [])
                self._occlusion_state = target.get("occlusion_state", "VISIBLE")
                self._hypotheses = target.get("hypotheses", [])
                self._search_sectors = target.get("search_sectors", [])
                self._merge_partners = target.get("merge_group", [])
                self._target_priority_score = target.get("priority_score", 0.0)
                self._target_behavior_pattern = target.get("behavior_pattern", "")

                # State transitions on re-acquisition
                if self._state in (TargetState.SEARCHING, TargetState.LOST, TargetState.SUSPECT):
                    self._state = TargetState.REACQUIRED
                    self._lost_time = 0.0
                    logger.info("Follow: reacquired {}", self._target_id)
                elif self._state == TargetState.CANDIDATE:
                    if identity_sim > self._identity_threshold:
                        self._state = TargetState.REACQUIRED
                        self._lost_time = 0.0
                elif self._state == TargetState.REACQUIRED:
                    self._state = TargetState.LOCKED
                elif self._state == TargetState.UNLOCKED:
                    pass
                else:
                    self._state = TargetState.LOCKED

                self._identity_similarity = identity_sim if identity_sim > 0 else 1.0

                error_x = cx - (frame_w / 2.0)
                self._last_error_x = error_x
                if abs(error_x) > 10:
                    self._search_direction = 1.0 if error_x > 0 else -1.0
                self._current_yaw_rate = self._pid.update(error_x)

                if self._follow_mode in (FollowMode.YAW_XY, FollowMode.INTERCEPT, FollowMode.STANDOFF):
                    self._compute_xy_velocity(error_x, bbox_h, frame_h)
                elif self._follow_mode == FollowMode.MANUAL_AIM:
                    self._current_vx = self._manual_vx
                    self._current_vy = self._manual_vy
                    self._current_vz = self._manual_vz
                else:
                    self._current_vx = 0.0
                    self._current_vy = 0.0
                    self._current_vz = 0.0

            else:
                # Target lost — hysteresis before full search
                if self._follow_mode == FollowMode.MANUAL_AIM:
                    self._current_vx = self._manual_vx
                    self._current_vy = self._manual_vy
                    self._current_vz = self._manual_vz
                else:
                    self._current_vx = 0.0
                    self._current_vy = 0.0
                    self._current_vz = 0.0

                if self._follow_mode == FollowMode.STANDOFF:
                    self._standoff_phase = "HOLD"

                if self._state in (TargetState.LOCKED, TargetState.REACQUIRED):
                    # First frame without target — enter SUSPECT
                    self._state = TargetState.SUSPECT
                    self._lost_time = now
                    logger.debug("Follow: target {} suspect (frame miss)", self._target_id)
                elif self._state == TargetState.SUSPECT:
                    lost_s = now - self._lost_time if self._lost_time > 0 else 0
                    if lost_s >= self._search_timeout_s:
                        self._state = TargetState.SEARCHING
                        logger.debug("Follow: target {} lost, searching ({:.1f}s)", self._target_id, lost_s)
                elif self._state == TargetState.CANDIDATE:
                    self._state = TargetState.SUSPECT
                    if self._lost_time == 0:
                        self._lost_time = now

    def _compute_identity_similarity(self, detail: dict) -> float:
        """Multi-signal identity matching using HSV vectors and descriptor overlap."""
        if not self._target_identity_memory and not self._identity_memory_upper_hsv:
            return 0.0
        if not detail:
            return 0.0

        current_desc = f"{detail.get('upper_color', '?')}/{detail.get('lower_color', '?')}"

        # Exact descriptor match
        if self._target_identity_memory and current_desc == self._target_identity_memory:
            return 0.9

        # HSV vector proximity (most discriminative signal)
        target_upper = detail.get("upper_hsv", [])
        target_lower = detail.get("lower_hsv", [])
        score = 0.0

        if target_upper and self._identity_memory_upper_hsv:
            h_diff = min(abs(target_upper[0] - self._identity_memory_upper_hsv[0]),
                         180 - abs(target_upper[0] - self._identity_memory_upper_hsv[0]))
            s_diff = abs(target_upper[1] - self._identity_memory_upper_hsv[1])
            hue_sim = max(0.0, 1.0 - h_diff / 30.0)
            sat_sim = max(0.0, 1.0 - s_diff / 128.0)
            upper_score = 0.6 * hue_sim + 0.3 * sat_sim
            score = max(score, 0.3 + 0.5 * upper_score)

        if target_lower and self._identity_memory_lower_hsv:
            h_diff = min(abs(target_lower[0] - self._identity_memory_lower_hsv[0]),
                         180 - abs(target_lower[0] - self._identity_memory_lower_hsv[0]))
            s_diff = abs(target_lower[1] - self._identity_memory_lower_hsv[1])
            hue_sim = max(0.0, 1.0 - h_diff / 30.0)
            sat_sim = max(0.0, 1.0 - s_diff / 128.0)
            lower_score = 0.6 * hue_sim + 0.3 * sat_sim
            score = max(score, 0.2 + 0.4 * lower_score)

        # Descriptor name overlap (fallback)
        if self._target_identity_memory and score < 0.3:
            parts_target = set(self._target_identity_memory.split("/"))
            parts_current = set(current_desc.split("/"))
            overlap = len(parts_target & parts_current)
            if overlap > 0:
                score = max(score, 0.5 + 0.2 * overlap)

        return score

    def _compute_xy_velocity(self, error_x: float, bbox_h: float, frame_h: int) -> None:
        if self._target_confidence < self._min_confidence:
            self._current_vx = 0.0
            self._current_vy = 0.0
            self._current_vz = 0.0
            return

        if not self._geofence_ok:
            self._current_vx = 0.0
            self._current_vy = 0.0
            self._current_vz = 0.0
            return

        # ---- STANDOFF MODE ----
        if self._follow_mode == FollowMode.STANDOFF:
            self._compute_standoff_velocity(error_x, bbox_h, frame_h)
            return

        # ---- INTERCEPT MODE ----
        if self._follow_mode == FollowMode.INTERCEPT:
            if bbox_h >= self._standoff_bbox_h:
                self._standoff_phase = "ORBITING"
                self._current_vx = 0.0
                self._current_vy = 0.0
                return
            self._standoff_phase = "APPROACHING"
            size_error = self._standoff_bbox_h - bbox_h
            vx = min(self._fwd_kp * size_error, 0.5)
            self._current_vx = vx
            self._current_vy = max(-0.3, min(0.3, self._lat_kp * error_x)) if abs(error_x) < self._yaw_align_threshold_px else 0.0
            return

        # ---- NORMAL yaw_xy ----
        size_error = self._target_height_px - bbox_h
        self._last_size_error = size_error
        vx = self._fwd_kp * size_error
        vy = self._lat_kp * error_x if abs(error_x) < self._yaw_align_threshold_px else 0.0

        vx = max(-self._max_vxy, min(self._max_vxy, vx))
        vy = max(-self._max_vxy, min(self._max_vxy, vy))
        if self._target_cy is not None and self._target_cy < frame_h * 0.2 and vx > 0:
            vx = 0.0
        if bbox_h > self._target_height_px * 2.0 and vx > 0:
            vx = 0.0
        self._current_vx = vx
        self._current_vy = vy

    def _get_actual_standoff_distance_m(self) -> float:
        """Return metric distance to target if world position + telemetry available, else 0.0."""
        if self._world_lat is None or self._world_lon is None:
            return 0.0
        if self._telemetry_cache is not None:
            pos, _ = self._telemetry_cache.get_position()
            if pos is not None:
                return _haversine_m(pos.lat, pos.lon, self._world_lat, self._world_lon)
        if self._backend is not None:
            try:
                telem = self._backend.get_telemetry()
                if telem.position is not None:
                    return _haversine_m(
                        telem.position.lat, telem.position.lon,
                        self._world_lat, self._world_lon,
                    )
            except Exception:
                pass
        return 0.0

    def _compute_standoff_velocity(self, error_x: float, bbox_h: float, frame_h: int) -> None:
        """Compute safe standoff velocities. Never commands collision course.

        Priority:
          1. Metric distance from world projection + telemetry (most accurate)
          2. Bbox height fallback (when metric unavailable)
        """
        actual_dist = self._get_actual_standoff_distance_m()
        use_metric = actual_dist > 0.0

        if use_metric:
            error_dist = actual_dist - self._standoff_distance_m
            deadband = self._standoff_deadband_m

            logger.debug(
                "standoff_logic_metric: dist={:.1f}m tgt={:.1f}m error={:.1f}m deadband={:.1f}m",
                actual_dist, self._standoff_distance_m, error_dist, deadband,
            )

            if error_dist < -deadband:
                # TOO CLOSE — back away
                self._standoff_phase = "RETREAT"
                vx = max(-self._standoff_approach_speed, self._standoff_metric_kp * error_dist)
                logger.info(
                    "standoff RETREAT (metric): dist={:.1f}m vx={:.3f} m/s",
                    actual_dist, vx,
                )
                self._current_vx = vx
                self._current_vy = 0.0
                self._current_vz = 0.0
                return

            if abs(error_dist) <= deadband:
                # WITHIN DEADBAND — hold or orbit
                if self._orbit_on_arrival:
                    self._standoff_phase = "ORBITING"
                else:
                    self._standoff_phase = "SAFE"
                logger.info(
                    "standoff HOLD/ORBIT (metric): dist={:.1f}m (tgt={:.1f}m ±{:.1f}m)",
                    actual_dist, self._standoff_distance_m, deadband,
                )
                self._current_vx = 0.0
                self._current_vy = 0.0
                self._current_vz = 0.0
                return

            # TOO FAR — approach
            self._standoff_phase = "APPROACHING"
            vx = min(self._standoff_metric_kp * error_dist, self._standoff_approach_speed)
            vx = max(0.0, min(self._max_vxy, vx))

            # Lateral: only when yaw aligned
            if abs(error_x) < self._yaw_align_threshold_px:
                vy = max(-0.3, min(0.3, self._lat_kp * error_x * 0.5))
            else:
                vy = 0.0

            self._current_vz = 0.0
            if self._target_cy is not None and self._target_cy < frame_h * 0.25:
                self._current_vz = 0.2

            logger.info(
                "standoff APPROACH (metric): dist={:.1f}m error={:.1f}m vx={:.3f} vy={:.3f} vz={:.3f} m/s",
                actual_dist, error_dist, vx, vy, self._current_vz,
            )
            self._current_vx = vx
            self._current_vy = vy
            return

        # ---- BBOX FALLBACK ----
        at_standoff = bbox_h >= self._standoff_bbox_h * (1.0 - self._standoff_hold_zone)
        too_close = bbox_h > self._standoff_bbox_h * 1.3

        logger.debug(
            "standoff_logic_bbox: bbox_h={:.1f} standoff_bbox_h={:.1f} at_standoff={} too_close={}",
            bbox_h, self._standoff_bbox_h, at_standoff, too_close,
        )

        if too_close:
            self._standoff_phase = "RETREAT"
            overshoot = bbox_h - self._standoff_bbox_h
            vx = -min(self._fwd_kp * overshoot, self._standoff_approach_speed)
            logger.info(
                "standoff RETREAT (bbox): overshoot={:.1f}px vx={:.3f} m/s",
                overshoot, vx,
            )
            self._current_vx = vx
            self._current_vy = 0.0
            self._current_vz = 0.0
            return

        if at_standoff:
            if self._orbit_on_arrival:
                self._standoff_phase = "ORBITING"
            else:
                self._standoff_phase = "SAFE"
            logger.info(
                "standoff HOLD/ORBIT (bbox): bbox_h={:.1f}px (target >= {:.1f}px)",
                bbox_h, self._standoff_bbox_h * (1.0 - self._standoff_hold_zone),
            )
            self._current_vx = 0.0
            self._current_vy = 0.0
            self._current_vz = 0.0
            return

        # APPROACHING
        self._standoff_phase = "APPROACHING"
        size_error = self._standoff_bbox_h - bbox_h
        vx = min(self._fwd_kp * size_error, self._standoff_approach_speed)
        vx = max(0.0, min(self._max_vxy, vx))

        if abs(error_x) < self._yaw_align_threshold_px:
            vy = max(-0.3, min(0.3, self._lat_kp * error_x * 0.5))
        else:
            vy = 0.0

        self._current_vz = 0.0
        if self._target_cy is not None and self._target_cy < frame_h * 0.25:
            self._current_vz = 0.2

        logger.info(
            "standoff APPROACH (bbox): size_error={:.1f}px vx={:.3f} vy={:.3f} vz={:.3f} m/s",
            size_error, vx, vy, self._current_vz,
        )
        self._current_vx = vx
        self._current_vy = vy

    def _record_home_position(self) -> None:
        if self._telemetry_cache is not None:
            pos, _ = self._telemetry_cache.get_position()
            if pos is not None:
                self._home_lat = pos.lat
                self._home_lon = pos.lon
                logger.info("Follow: home ({:.6f}, {:.6f})", self._home_lat, self._home_lon)
            return
        try:
            telem = self._backend.get_telemetry()
            if telem.position is not None:
                self._home_lat = telem.position.lat
                self._home_lon = telem.position.lon
                logger.info("Follow: home ({:.6f}, {:.6f})", self._home_lat, self._home_lon)
        except Exception:
            logger.warning("Follow: could not read home position")

    def _check_geofence(self) -> None:
        if self._home_lat is None or self._home_lon is None:
            self._geofence_ok = True
            self._distance_from_home = 0.0
            return
        if self._telemetry_cache is not None:
            pos, _ = self._telemetry_cache.get_position()
            if pos is None:
                return
            dist = _haversine_m(
                self._home_lat, self._home_lon, pos.lat, pos.lon,
            )
            was_ok = self._geofence_ok
            self._geofence_ok = dist < self._geofence_radius_m
            self._distance_from_home = dist
            if was_ok and not self._geofence_ok:
                logger.warning("Follow: GEOFENCE breach at {:.1f}m (limit {:.0f}m)",
                               dist, self._geofence_radius_m)
            return
        try:
            telem = self._backend.get_telemetry()
            if telem.position is None:
                return
            dist = _haversine_m(
                self._home_lat, self._home_lon,
                telem.position.lat, telem.position.lon,
            )
            self._distance_from_home = dist
            was_ok = self._geofence_ok
            self._geofence_ok = dist < self._geofence_radius_m
            if was_ok and not self._geofence_ok:
                logger.warning("Follow: GEOFENCE breach at {:.1f}m (limit {:.0f}m)",
                               dist, self._geofence_radius_m)
        except Exception:
            pass

    def get_diagnostics(self) -> FollowDiagnostics:
        with self._lock:
            lost_s = 0.0
            if self._state in (TargetState.SEARCHING, TargetState.LOST) and self._lost_time > 0:
                lost_s = time.monotonic() - self._lost_time

            standoff_ok = self._last_bbox_height >= self._standoff_bbox_h * 0.85

            # Compute actual standoff distance from world positions if available
            standoff_actual_m = 0.0
            if self._world_lat is not None and self._world_lon is not None:
                if self._telemetry_cache is not None:
                    pos, _ = self._telemetry_cache.get_position()
                    if pos is not None:
                        standoff_actual_m = _haversine_m(
                            pos.lat, pos.lon, self._world_lat, self._world_lon,
                        )
                elif self._backend is not None:
                    try:
                        telem = self._backend.get_telemetry()
                        if telem.position is not None:
                            standoff_actual_m = _haversine_m(
                                telem.position.lat, telem.position.lon,
                                self._world_lat, self._world_lon,
                            )
                    except Exception:
                        pass

            return FollowDiagnostics(
                state=self._state,
                follow_mode=self._follow_mode,
                target_id=self._target_id or "",
                identity_descriptor=self._target_identity_desc,
                identity_similarity=self._identity_similarity,
                error_x=self._last_error_x,
                error_y=0.0,
                yaw_rate=self._current_yaw_rate,
                vx=self._current_vx,
                vy=self._current_vy,
                vz=self._current_vz,
                pid_state=self._pid.state,
                offboard_active=self._offboard_engaged,
                frames_since_update=int(time.monotonic() - self._tracker_time) if self._tracker_time > 0 else 0,
                target_bbox=list(self._target_bbox) if self._target_bbox else None,
                target_speed=self._target_speed,
                target_confidence=self._target_confidence,
                search_direction=self._search_direction,
                target_lost_s=lost_s,
                bbox_height_px=self._last_bbox_height,
                size_error=self._last_size_error,
                geofence_ok=self._geofence_ok,
                distance_from_home_m=self._distance_from_home,
                standoff_distance_ok=standoff_ok,
                standoff_phase=self._standoff_phase,
                standoff_target_m=self._standoff_distance_m,
                standoff_altitude_m=self._standoff_altitude_m,
                standoff_actual_m=standoff_actual_m,
                deterrence_active=self._deterrence_marker and standoff_ok,
                world_position={"lat": self._world_lat, "lon": self._world_lon} if self._world_lat else None,
                world_velocity={"speed_mps": self._world_speed_mps, "heading_rad": self._world_heading_rad} if self._world_speed_mps > 0 else None,
                occlusion_state=getattr(self, "_occlusion_state", ""),
                emergence_points=list(self._emergence_points),
                hypotheses=list(self._hypotheses),
                best_hypothesis=(max(self._hypotheses, key=lambda h: h.get("probability", 0)).get("type", "")
                                if self._hypotheses else ""),
                merge_group_partners=list(self._merge_partners),
                search_sectors=list(self._search_sectors),
                priority_score=self._target_priority_score,
                behavior_pattern=self._target_behavior_pattern,
            )

    # ------------------------------------------------------------------
    # Background follow loop
    # ------------------------------------------------------------------

    def _start_loop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._follow_loop, daemon=True)
        self._thread.start()

    def _stop_loop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._disengage_offboard()

    def _follow_loop(self) -> None:
        interval = 1.0 / self._offboard_hz
        logger.info("Follow loop started at {} Hz (mode={})", self._offboard_hz, self._follow_mode.value)

        if not self._engage_offboard():
            logger.error("Follow: failed to engage offboard, aborting follow loop (vision loop continues)")
            with self._lock:
                self._state = TargetState.UNLOCKED
                self._offboard_engaged = False
            return

        geofence_counter = 0
        log_counter = 0

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            with self._lock:
                state = self._state
                yaw_rate = self._current_yaw_rate
                vx = self._manual_vx if self._follow_mode == FollowMode.MANUAL_AIM else self._current_vx
                vy = self._manual_vy if self._follow_mode == FollowMode.MANUAL_AIM else self._current_vy
                vz = self._manual_vz if self._follow_mode == FollowMode.MANUAL_AIM else self._current_vz
                search_dir = self._search_direction
                tracker_age = time.monotonic() - self._tracker_time if self._tracker_time > 0 else 999.0
                lost_time = self._lost_time
                standoff_phase = self._standoff_phase
                bbox_h = self._last_bbox_height
                world_lat = self._world_lat
                world_lon = self._world_lon

            # Periodic distance / command logging (every ~2 s)
            log_counter += 1
            if log_counter >= int(self._offboard_hz * 2.0):
                log_counter = 0
                if self._follow_mode == FollowMode.STANDOFF and world_lat is not None:
                    actual_dist = self._get_actual_standoff_distance_m()
                    mode_str = "metric" if actual_dist > 0.0 else "bbox"
                    logger.info(
                        "standoff_trend: mode={} dist={:.1f}m (tgt={:.1f}m) phase={} bbox_h={:.1f}px cmd=[{:.2f},{:.2f},{:.2f}]",
                        mode_str, actual_dist, self._standoff_distance_m, standoff_phase, bbox_h, vx, vy, vz,
                    )

            geofence_counter += 1
            if geofence_counter >= int(self._offboard_hz):
                geofence_counter = 0
                self._check_geofence()

            if tracker_age > 0.3:
                yaw_rate *= self._decay_rate
                if self._follow_mode != FollowMode.MANUAL_AIM:
                    vx *= self._decay_rate
                    vy *= self._decay_rate
                    vz *= self._decay_rate

            if state == TargetState.UNLOCKED:
                break

            elif state == TargetState.SUSPECT:
                # Brief target miss — hold last command with gentle decay
                decay = 0.95
                yaw_rate *= decay
                if self._follow_mode == FollowMode.MANUAL_AIM:
                    vx = self._manual_vx
                    vy = self._manual_vy
                    vz = self._manual_vz
                else:
                    vx *= decay
                    vy *= decay
                    vz *= decay

            elif state == TargetState.SEARCHING:
                now = time.monotonic()
                lost_s = now - lost_time if lost_time > 0 else 0

                # Merge-group slowdown: wait for split
                merged_prob = 0.0
                for h in self._hypotheses:
                    if h.get("type") == "MERGED_GROUP":
                        merged_prob = h.get("probability", 0)
                if merged_prob > 0.3 and self._merge_partners:
                    yaw_rate = self._search_yaw_rate * search_dir * 0.5
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                    # Skip other search logic
                elif self._search_sectors:
                    # Hypothesis-weighted search using sectors
                    try:
                        if self._telemetry_cache is not None:
                            drone_heading, _ = self._telemetry_cache.get_heading()
                            drone_heading = drone_heading or 0.0
                        else:
                            tele = self._backend.get_telemetry()
                            drone_heading = tele.heading_deg
                        # Compute weighted bearing from sectors
                        weighted_bearing = 0.0
                        total_weight = 0.0
                        for sec in self._search_sectors:
                            w = sec.get("probability", 0)
                            weighted_bearing += sec["bearing_deg"] * w
                            total_weight += w
                        if total_weight > 0:
                            weighted_bearing /= total_weight
                        heading_diff = ((weighted_bearing - drone_heading + 180) % 360) - 180
                        if abs(heading_diff) > 5:
                            yaw_rate = self._search_yaw_rate * (1.0 if heading_diff > 0 else -1.0) * 0.6
                        else:
                            yaw_rate = self._search_yaw_rate * search_dir
                    except Exception:
                        yaw_rate = self._search_yaw_rate * search_dir
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                elif self._emergence_points and self._world_lat is not None:
                    # Fallback: use highest-probability emergence point
                    best_ep = max(self._emergence_points, key=lambda e: e.get("probability", 0))
                    ep_lat = best_ep.get("lat", 0)
                    ep_lon = best_ep.get("lon", 0)
                    try:
                        if self._telemetry_cache is not None:
                            drone_heading, _ = self._telemetry_cache.get_heading()
                            drone_heading = drone_heading or 0.0
                            pos, _ = self._telemetry_cache.get_position()
                            drone_lat = pos.lat if pos else 0.0
                            drone_lon = pos.lon if pos else 0.0
                        else:
                            tele = self._backend.get_telemetry()
                            drone_lat = tele.position.lat
                            drone_lon = tele.position.lon
                            drone_heading = tele.heading_deg
                        dn = (ep_lat - drone_lat) * 111320.0
                        de = (ep_lon - drone_lon) * 111320.0 * math.cos(math.radians(drone_lat))
                        target_bearing = math.degrees(math.atan2(de, dn))
                        heading_diff = ((target_bearing - drone_heading + 180) % 360) - 180
                        if abs(heading_diff) > 10:
                            yaw_rate = self._search_yaw_rate * (1.0 if heading_diff > 0 else -1.0) * 0.6
                        else:
                            yaw_rate = self._search_yaw_rate * search_dir
                    except Exception:
                        yaw_rate = self._search_yaw_rate * search_dir
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                elif self._last_known_position is not None:
                    # Pixel-space fallback: rotate toward predicted target position
                    pred_cx, pred_cy = self._last_known_position
                    if self._last_known_heading != 0.0:
                        speed_px = self._target_speed * (0.95 ** (lost_s * self._offboard_hz))
                        pred_cx += speed_px * math.cos(self._last_known_heading) * lost_s * 5
                        pred_cy += speed_px * math.sin(self._last_known_heading) * lost_s * 5
                    error_x = pred_cx - (self._frame_w / 2.0)
                    if abs(error_x) > 30:
                        yaw_rate = self._pid.update(error_x) * 0.4
                    else:
                        yaw_rate = self._search_yaw_rate * search_dir
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                else:
                    yaw_rate = self._search_yaw_rate * search_dir
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

                if self._follow_mode == FollowMode.MANUAL_AIM:
                    vx = self._manual_vx
                    vy = self._manual_vy
                    vz = self._manual_vz
                else:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

                # Priority-aware timeout: high-priority targets get extended search
                effective_timeout = self._lost_timeout_s
                if self._target_priority_score > 0.5:
                    effective_timeout = self._lost_timeout_s * (1.0 + self._target_priority_score)

                if lost_s > effective_timeout:
                    logger.warning("Follow: target lost {:.1f}s, disengaging", lost_s)
                    with self._lock:
                        self._state = TargetState.LOST
                    break

            elif state == TargetState.CANDIDATE:
                yaw_rate = self._search_yaw_rate * 0.5 * search_dir
                if self._follow_mode == FollowMode.MANUAL_AIM:
                    vx = self._manual_vx
                    vy = self._manual_vy
                    vz = self._manual_vz
                else:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

            elif state == TargetState.LOST:
                yaw_rate *= self._decay_rate
                if self._follow_mode == FollowMode.MANUAL_AIM:
                    vx = self._manual_vx
                    vy = self._manual_vy
                    vz = self._manual_vz
                else:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

            elif state == TargetState.LOCKED:
                if abs(yaw_rate) < self._min_yaw_rate and abs(self._last_error_x) > 20:
                    yaw_rate = self._min_yaw_rate * (1.0 if self._last_error_x > 0 else -1.0)

                if not self._geofence_ok:
                    vx = 0.0
                    vy = 0.0
                    if self._distance_from_home > 0:
                        yaw_rate *= 0.5

                # Intercept: orbit when at standoff
                if self._follow_mode == FollowMode.INTERCEPT and standoff_phase == "ORBITING":
                    yaw_rate = self._orbit_yaw_rate
                    vx = 0.0
                    vy = 0.0

                # Standoff: orbit on arrival
                if self._follow_mode == FollowMode.STANDOFF and standoff_phase == "ORBITING":
                    yaw_rate = self._orbit_yaw_rate
                    vx = 0.0
                    vy = 0.0

            # Altitude safety: clamp vz when altitude is out of bounds
            if self._telemetry_cache is not None:
                pos, _ = self._telemetry_cache.get_position()
                if pos is not None and pos.alt_m is not None:
                    # Enforce hard limits: 5.0m to 25.0m
                    if pos.alt_m >= 25.0 and vz > 0.0:
                        vz = 0.0
                    elif pos.alt_m <= 5.0 and vz < 0.0:
                        vz = 0.0

            # Final safety clamp
            yaw_rate = max(-self._max_yaw_rate, min(self._max_yaw_rate, yaw_rate))
            vx = max(-self._max_vxy, min(self._max_vxy, vx))
            vy = max(-self._max_vxy, min(self._max_vxy, vy))
            vz = max(-1.0, min(1.0, vz))

            # Standoff extra safety: never allow forward speed > approach_speed
            if self._follow_mode == FollowMode.STANDOFF and vx > self._standoff_approach_speed:
                vx = self._standoff_approach_speed

            try:
                if self._command_slot is not None:
                    self._write_velocity_command(
                        vx, vy, vz, yaw_rate, f"follow_{self._follow_mode.value}",
                    )
                else:
                    self._backend.offboard_set_velocity_body(vx, vy, vz, yaw_rate)
            except Exception as exc:
                logger.error("Follow: offboard setpoint failed: {}", exc)
                break

            with self._lock:
                self._current_yaw_rate = yaw_rate
                self._current_vx = vx
                self._current_vy = vy
                self._current_vz = vz

            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

        self._disengage_offboard()
        with self._lock:
            if self._state != TargetState.UNLOCKED:
                self._state = TargetState.UNLOCKED
        logger.info("Follow loop stopped")

    def _engage_offboard(self) -> bool:
        if self._command_slot is not None:
            self._offboard_engaged = True
            return True
        try:
            result = self._backend.offboard_start()
            if result.success:
                self._offboard_engaged = True
                logger.info("Follow: offboard engaged")
                return True
            logger.error("Follow: offboard rejected: {}", result.message)
            return False
        except Exception as exc:
            logger.error("Follow: offboard start exception: {}", exc)
            return False

    def _disengage_offboard(self) -> None:
        if self._command_slot is not None:
            self._offboard_engaged = False
            self._emit_zero_command("follow_disengage")
            return
        if not self._offboard_engaged:
            return
        try:
            self._backend.offboard_stop()
            logger.info("Follow: offboard disengaged, holding position")
        except Exception as exc:
            logger.error("Follow: offboard stop failed: {}", exc)
            try:
                self._backend.hold()
            except Exception:
                pass
        self._offboard_engaged = False

    def emergency_stop(self) -> None:
        logger.warning("Follow: EMERGENCY STOP")
        with self._lock:
            self._state = TargetState.UNLOCKED
            self._target_id = None
            self._current_yaw_rate = 0.0
            self._current_vx = 0.0
            self._current_vy = 0.0
            self._current_vz = 0.0
            self._manual_vx = 0.0
            self._manual_vy = 0.0
            self._manual_vz = 0.0
        self._stop_loop()
        self._emit_zero_command("follow_emergency_stop")
