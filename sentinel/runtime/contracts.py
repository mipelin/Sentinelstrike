"""Runtime data contracts for the blackboard bus."""

from __future__ import annotations

from pydantic import BaseModel


class FrameSnapshot(BaseModel):
    """Latest captured frame metadata. Pixel data stays in LatestSlot[np.ndarray]."""

    frame_id: int = 0
    seq: int = 0
    ts_monotonic: float = 0.0
    ts_utc: str = ""
    shape: tuple[int, int, int] | None = None  # (h, w, channels)


class DetectionSet(BaseModel):
    """Detections produced by perception for one frame."""

    frame_id: int = 0
    seq: int = 0
    ts_monotonic: float = 0.0
    detections: list[dict] = []
    backend: str = "unknown"
    inference_ms: float = 0.0


class TrackSet(BaseModel):
    """Tracks produced by tracker for one frame."""

    frame_id: int = 0
    seq: int = 0
    ts_monotonic: float = 0.0
    tracks: list[dict] = []
    tracker_type: str = "unknown"
    active_count: int = 0
    lost_count: int = 0


class WorldModelSnapshot(BaseModel):
    """Aggregated world model from ISR tracker."""

    frame_id: int = 0
    seq: int = 0
    ts_monotonic: float = 0.0
    active_tracks: list[dict] = []
    hypotheses: list[dict] = []
    behavior_events: list[dict] = []
    world_positions: list[dict] = []


class DroneState(BaseModel):
    """Current drone state from telemetry."""

    seq: int = 0
    ts_monotonic: float = 0.0
    ts_utc: str = ""
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    position: dict | None = None  # {lat, lon, alt_m}
    heading_deg: float | None = None
    groundspeed_mps: float | None = None
    battery_pct: float | None = None


class VelocityCommand(BaseModel):
    """Offboard velocity command."""

    ts_monotonic: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yawspeed: float = 0.0
    source: str = "unknown"


class FlightStatus(BaseModel):
    """High-level flight status."""

    ts_monotonic: float = 0.0
    ts_utc: str = ""
    state: str = "UNKNOWN"  # DISARMED, ARMED, TAKEOFF, IN_FLIGHT, LANDING, RTL
    offboard_active: bool = False
    follow_mode: str = "none"
    follow_target: str | None = None
