"""Configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SystemConfig(BaseModel):
    mode: str = "SIMULATION_MODE"
    vehicle_id: str = "uav_001"
    mission_id: str = "demo_001"
    log_level: str = "INFO"


class RecorderConfig(BaseModel):
    enabled: bool = True
    output_dir: str = "runs"
    record_video: bool = False


class SimulationConfig(BaseModel):
    link_loss_enabled: bool = False
    link_loss_start_after_s: int = 60
    link_loss_duration_s: int = 45


class MavlinkConfig(BaseModel):
    enabled: bool = False
    backend: str = "mock"
    connection_url: str = "udpin://0.0.0.0:14540"
    allow_arm: bool = False
    allow_takeoff: bool = False
    allow_real_backend: bool = False
    max_takeoff_altitude_m: float = 30
    max_goto_distance_m: float = 1000
    connect_timeout_s: int = 15
    default_waypoint_altitude_m: float = 50
    default_speed_mps: float = 10


class PerceptionConfig(BaseModel):
    enabled: bool = False
    backend: str = "mock"
    model_path: str = "yolov8n.pt"
    source: str = "data/videos/demo.mp4"
    output_annotated_video: bool = False
    confidence_threshold: float = 0.35
    frame_stride: int = 1
    max_frames: int | None = 300
    classes: list[str] = ["person", "car", "truck", "bus"]
    save_detections_jsonl: bool = True
    device: str = "cpu"
    image_size: int = 640


class TrackerConfig(BaseModel):
    enabled: bool = False
    algorithm: str = "simple_iou"
    iou_threshold: float = 0.3
    max_lost_frames: int = 15
    min_confidence: float = 0.0
    save_tracks_jsonl: bool = True
    prediction_enabled: bool = False
    distance_threshold_px: float = 100.0
    iou_weight: float = 1.0
    distance_weight: float = 0.0
    suppress_stationary_after_s: float = Field(default=10.0, ge=0)
    stationary_speed_threshold_px_s: float = Field(default=3.0, ge=0)
    fps: float = Field(default=5.0, ge=0.1)
    min_hits_to_confirm: int = Field(default=1, ge=1)
    publish_tentative_tracks: bool = False
    fast_confirm_confidence: float = Field(default=0.85, ge=0, le=1)
    reacquire_window_frames: int = Field(default=20, ge=1)
    duplicate_distance_px: float = Field(default=80.0, ge=0)
    duplicate_iou_threshold: float = Field(default=0.1, ge=0, le=1)
    class_aware_matching: bool = True
    confidence_ema_alpha: float = Field(default=0.35, ge=0, le=1)
    bbox_smoothing_enabled: bool = False
    bbox_ema_alpha: float = Field(default=0.35, ge=0, le=1)


class GeolocalizerConfig(BaseModel):
    enabled: bool = False
    method: str = "flat_ground_pinhole_v1"
    assumed_ground_alt_m: float = 0.0
    default_accuracy_estimate_m: float = Field(default=35.0, ge=0)
    min_track_confidence: float = Field(default=0.0, ge=0, le=1)
    save_geo_observations_jsonl: bool = True
    camera: dict = {}
    publish_active_only: bool = True
    min_publish_interval_s: float = Field(default=2.0, ge=0)
    min_movement_m: float = Field(default=5.0, ge=0)
    max_observations_per_track: int = Field(default=50, ge=1)

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        if v != "flat_ground_pinhole_v1":
            raise ValueError(f"Unsupported geolocalizer method: {v}")
        return v


class TakConfig(BaseModel):
    enabled: bool = False
    mode: str = "dry_run"
    cot_host: str = "127.0.0.1"
    cot_port: int = Field(default=8087, ge=1, le=65535)
    callsign: str = "ONS-UAV-001"
    team: str = "Cyan"
    role: str = "UAV"
    stale_after_s: int = Field(default=120, gt=0)
    save_tak_messages_jsonl: bool = True
    publish_vehicle_every_s: float = Field(default=5.0, ge=0)
    publish_track_every_s: float = Field(default=3.0, ge=0)
    max_messages_per_run: int = Field(default=1000, ge=1)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in ("dry_run", "udp"):
            raise ValueError(f"Unsupported TAK mode: {v} (expected dry_run or udp)")
        return v

    @field_validator("callsign")
    @classmethod
    def _validate_callsign(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("callsign must not be empty")
        return v


class SitlLocalMissionConfig(BaseModel):
    enabled: bool = True
    size_m: float = Field(default=80.0, gt=0)
    altitude_m: float = Field(default=30.0, gt=0)
    spacing_m: float = Field(default=30.0, gt=0)
    max_range_km: float = Field(default=2.0, gt=0)
    default_speed_mps: float = Field(default=5.0, gt=0)
    max_waypoint_distance_m: float = Field(default=500.0, gt=0)


class OperatorConfig(BaseModel):
    enabled: bool = False
    operator_id: str = "local_operator"
    simulator_mode: str = "confirm_above_threshold"
    min_track_confidence_for_confirmation: float = 0.6
    min_observation_confidence_for_orbit: float = 0.6
    require_human_for_orbit: bool = True
    require_human_for_abort: bool = False
    auto_reject_below_confidence: bool = True


class SafetyConfig(BaseModel):
    enabled: bool = False
    abort_actions: list[str] = ["HOLD", "LAND"]
    return_to_safe_action: str = "RETURN_HOME"
    link_loss_action: str = "RETURN_HOME"
    low_battery_threshold_pct: float = 20.0
    low_battery_action: str = "RETURN_HOME"
    telemetry_stale_threshold: int = 5
    telemetry_stale_action: str = "HOLD"
    allow_disarm: bool = False


class VideoSourceConfig(BaseModel):
    source_type: str = "file"
    file_path: str = "data/videos/demo.mp4"
    webcam_index: int = 0
    rtsp_url: str = ""
    gazebo_camera_topic: str = ""
    gazebo_world_name: str = "default"
    reconnect_enabled: bool = True
    reconnect_interval_s: float = 5.0
    frame_width: int | None = None
    frame_height: int | None = None
    target_fps: float = 5.0

    @field_validator("source_type")
    @classmethod
    def _validate_source_type(cls, v: str) -> str:
        if v not in ("file", "webcam", "rtsp", "gazebo_camera"):
            raise ValueError(f"Unsupported source_type: {v} (expected file, webcam, rtsp, or gazebo_camera)")
        return v


class RuntimeConfig(BaseModel):
    target_fps: float = Field(default=10.0, ge=1.0, le=60.0)
    max_frames: int | None = None
    save_frames: bool = False
    save_latest_frame: bool = True
    save_annotated_every_n_frames: int = Field(default=10, ge=0)
    disk_budget_mb: int = Field(default=2048, ge=64)
    watchdog_sensor_stale_s: float = Field(default=5.0, ge=1.0)
    watchdog_telemetry_stale_s: float = Field(default=10.0, ge=1.0)
    watchdog_min_fps: float = Field(default=2.0, ge=0.5)
    watchdog_ram_threshold_pct: float = Field(default=90.0, ge=50.0, le=100.0)
    watchdog_disk_threshold_pct: float = Field(default=95.0, ge=50.0, le=100.0)
    frame_timeout_s: float = Field(default=5.0, ge=1.0)
    mavlink_telemetry_timeout_s: float = Field(default=3.0, ge=0.5)
    log_rotation_enabled: bool = True
    log_max_size_mb: int = Field(default=512, ge=1)


class AppConfig(BaseModel):
    system: SystemConfig = SystemConfig()
    recorder: RecorderConfig = RecorderConfig()
    simulation: SimulationConfig = SimulationConfig()
    mavlink: MavlinkConfig = MavlinkConfig()
    perception: PerceptionConfig = PerceptionConfig()
    tracker: TrackerConfig = TrackerConfig()
    geolocalizer: GeolocalizerConfig = GeolocalizerConfig()
    tak: TakConfig = TakConfig()
    sitl_local_mission: SitlLocalMissionConfig = SitlLocalMissionConfig()
    operator: OperatorConfig = OperatorConfig()
    safety: SafetyConfig = SafetyConfig()
    video: VideoSourceConfig = VideoSourceConfig()
    runtime: RuntimeConfig = RuntimeConfig()
