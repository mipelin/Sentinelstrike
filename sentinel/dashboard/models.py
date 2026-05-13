"""Dashboard Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel


class RunInfo(BaseModel):
    run_id: str
    path: str
    created_at: str = ""
    mission_id: str = ""
    has_report: bool = False
    has_summary: bool = False


class TimelineEvent(BaseModel):
    t: str
    source: str
    type: str
    severity: str = "info"
    label: str = ""
    payload: dict = {}


class WaypointLayer(BaseModel):
    lat: float
    lon: float
    label: str = ""


class ObservationLayer(BaseModel):
    lat: float
    lon: float
    track_id: str = ""
    class_name: str = ""
    confidence: float = 0.0
    confirmed: bool = False
    observation_id: str = ""
    frame_id: int | None = None


class TrackBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TargetCard(BaseModel):
    track_id: str
    class_name: str = ""
    confidence: float = 0.0
    age_frames: int = 0
    lost_frames: int = 0
    status: str = "ACTIVE"
    confirmation_state: str = "UNREVIEWED"
    operator_disposition: str = "PENDING"
    last_seen_utc: str = ""
    first_seen_utc: str = ""
    stale: bool = False
    reacquired_count: int = 0
    frame_id: int = 0
    bbox: TrackBox | None = None
    history_px: list[list[float]] = []
    observation_id: str | None = None
    observation_lat: float | None = None
    observation_lon: float | None = None
    movement_state: str = "STATIONARY"
    average_speed_px_s: float = 0.0
    displacement_px: float = 0.0
    stationary_frames: int = 0
    moving_frames: int = 0
    priority_score: float = 0.0
    priority_level: str = "LOW"
    suppressed: bool = False


class OperatorQueueItem(BaseModel):
    observation_id: str
    track_id: str = ""
    requested_at: str = ""
    policy_action: str = ""
    decision_state: str = "pending"
    reason: str = ""


class ReplayFrame(BaseModel):
    frame_id: int
    frame_name: str
    timestamp_utc: str = ""
    vehicle_state: dict | None = None
    targets: list[TargetCard] = []
    observations: list[ObservationLayer] = []


class SafetyEventLayer(BaseModel):
    lat: float | None = None
    lon: float | None = None
    action: str = ""
    trigger: str = ""


class UavPositionLayer(BaseModel):
    lat: float
    lon: float
    timestamp: str = ""
    frame_id: int | None = None
    heading: float | None = None
    alt_m: float | None = None
    speed_mps: float | None = None
    battery_pct: float | None = None
    mode: str = ""
    armed: bool = False
    stale: bool = False
    vehicle_id: str = ""


class UavTrackPoint(BaseModel):
    lat: float
    lon: float
    alt_m: float | None = None
    timestamp_utc: str = ""
    frame_id: int | None = None


class MapLayers(BaseModel):
    waypoints: list[WaypointLayer] = []
    uav_positions: list[UavPositionLayer] = []
    uav_track: list[UavTrackPoint] = []
    observations: list[ObservationLayer] = []
    safety_events: list[SafetyEventLayer] = []
