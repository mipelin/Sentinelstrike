"""Shared Pydantic models for the ONS Sentinel system."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class GeoPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_m: float | None = None


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def _validate_coords(self) -> BoundingBox:
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self


class Detection(BaseModel):
    frame_id: int
    timestamp_utc: str
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: BoundingBox
    source: str = "unknown"
    detection_id: str | None = None
    metadata: dict = {}


class Track(BaseModel):
    track_id: str
    class_name: str
    confidence: float
    bbox_xyxy: BoundingBox | None = None
    status: str = "active"
    age_frames: int = 0
    last_seen_utc: str
    first_seen_utc: str | None = None
    lost_frames: int = 0
    source_detection_id: str | None = None
    metadata: dict = {}


class VehicleState(BaseModel):
    vehicle_id: str
    timestamp_utc: str
    position: GeoPoint | None = None
    heading_deg: float | None = Field(default=None, ge=0, le=360)
    groundspeed_mps: float | None = None
    mode: str = "UNKNOWN"
    armed: bool = False
    battery_pct: float | None = Field(default=None, ge=0, le=100)


class MissionConstraints(BaseModel):
    max_altitude_m: float = 120
    min_altitude_m: float = 40
    max_range_km: float = 5
    human_authorization_required: bool = True
    lost_link_behavior: str = "return_home"
    search_spacing_m: float = Field(default=50, gt=0)
    default_altitude_m: float = 80
    default_speed_mps: float = Field(default=10, gt=0)
    max_waypoints: int = Field(default=100, ge=3)

    @model_validator(mode="after")
    def _validate_altitudes(self) -> MissionConstraints:
        if not (self.min_altitude_m <= self.default_altitude_m <= self.max_altitude_m):
            raise ValueError(
                f"default_altitude_m ({self.default_altitude_m}) must be between "
                f"min_altitude_m ({self.min_altitude_m}) and max_altitude_m ({self.max_altitude_m})"
            )
        return self


class MissionRequest(BaseModel):
    mission_id: str
    launch_point: GeoPoint
    area_of_interest: list[GeoPoint]
    search_pattern: str = "lawnmower"
    constraints: MissionConstraints = Field(default_factory=MissionConstraints)

    @field_validator("area_of_interest")
    @classmethod
    def _min_three_points(cls, v: list[GeoPoint]) -> list[GeoPoint]:
        if len(v) < 3:
            raise ValueError("area_of_interest must have at least 3 points")
        return v


class MissionPlan(BaseModel):
    mission_id: str
    waypoints: list[GeoPoint]
    search_pattern: str
    abort_points: list[GeoPoint] = []
    return_home_point: GeoPoint
    operator_checkpoints: list[GeoPoint] = []
    total_distance_m: float | None = None
    estimated_duration_s: float | None = None
    planner_version: str = "mission_planner_v1"
    metadata: dict = {}


class MissionCommand(BaseModel):
    command_id: str
    mission_id: str
    command_type: str
    payload: dict = {}
    timestamp_utc: str


class OperatorDecision(BaseModel):
    decision_id: str
    mission_id: str
    timestamp_utc: str
    operator: str = "local"
    decision: str
    target_track_id: str | None = None


class CameraModel(BaseModel):
    camera_id: str = "cam_001"
    width_px: int = Field(default=640, gt=0)
    height_px: int = Field(default=480, gt=0)
    horizontal_fov_deg: float = Field(default=70.0, gt=0, lt=180)
    vertical_fov_deg: float = Field(default=45.0, gt=0, lt=180)
    pitch_deg: float = Field(default=-45.0, ge=-90, le=90)
    yaw_offset_deg: float = Field(default=0.0, ge=-180, le=180)
    roll_deg: float = Field(default=0.0, ge=-180, le=180)
    mounting: str = "fixed_forward_down"


class SensorFrame(BaseModel):
    frame_id: int
    capture_timestamp_monotonic: float
    capture_timestamp_utc: str
    receive_timestamp_monotonic: float
    source: str = "unknown"
    metadata: dict = {}


class GeoObservation(BaseModel):
    observation_id: str
    mission_id: str
    track_id: str
    class_name: str
    timestamp_utc: str
    estimated_location: GeoPoint
    accuracy_estimate_m: float = Field(default=35.0, ge=0)
    method: str = "flat_ground_pinhole_v1"
    confidence: float = Field(ge=0, le=1)
    source: str = "geolocalizer"
    gimbal_pitch_deg: float | None = None
    gimbal_yaw_deg: float | None = None
    telemetry_age_ms: float | None = None
    frame_to_telemetry_delta_ms: float | None = None
    metadata: dict = {}
