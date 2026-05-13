"""Unified sensor pose model for geolocalization and pipeline staging."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sentinel.common.types import CameraModel, GeoPoint, VehicleState


class SensorPose(BaseModel):
    """Complete sensor state at a point in time.

    Combines vehicle telemetry + camera intrinsics into a single snapshot
    that can be passed through the pipeline stages.
    """

    vehicle_id: str
    frame_timestamp_utc: str
    capture_timestamp_monotonic_s: float | None = None
    position: GeoPoint | None = None
    heading_deg: float | None = Field(default=None, ge=0, le=360)
    groundspeed_mps: float | None = None
    pitch_deg: float | None = Field(default=None, ge=-90, le=90)
    roll_deg: float | None = Field(default=None, ge=-180, le=180)
    yaw_deg: float | None = Field(default=None, ge=0, le=360)
    altitude_above_ground_m: float | None = None
    camera: CameraModel | None = None
    telemetry_age_ms: float | None = None
    frame_age_ms: float | None = None
    source: str = "unknown"
    stale: bool = False
    metadata: dict = {}


def build_sensor_pose(
    vehicle_state: VehicleState,
    frame_timestamp_utc: str,
    camera_model: CameraModel | None = None,
    capture_monotonic_s: float | None = None,
) -> SensorPose:
    """Build a SensorPose from a VehicleState + optional camera and timing info."""
    alt_agl: float | None = None
    if vehicle_state.position is not None and vehicle_state.position.alt_m is not None:
        alt_agl = vehicle_state.position.alt_m

    return SensorPose(
        vehicle_id=vehicle_state.vehicle_id,
        frame_timestamp_utc=frame_timestamp_utc,
        capture_timestamp_monotonic_s=capture_monotonic_s,
        position=vehicle_state.position,
        heading_deg=vehicle_state.heading_deg,
        groundspeed_mps=vehicle_state.groundspeed_mps,
        altitude_above_ground_m=alt_agl,
        camera=camera_model,
        source=vehicle_state.mode,
    )


def sensor_pose_to_vehicle_state(pose: SensorPose) -> VehicleState:
    """Convert a SensorPose back to a VehicleState for backward compat."""
    return VehicleState(
        vehicle_id=pose.vehicle_id,
        timestamp_utc=pose.frame_timestamp_utc,
        position=pose.position,
        heading_deg=pose.heading_deg,
        groundspeed_mps=pose.groundspeed_mps,
        mode=pose.source,
    )
