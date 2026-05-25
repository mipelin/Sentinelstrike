"""Lightweight ISR-grade pixel-to-world coordinate projection.

Projects image-space bounding boxes to world-space ground positions using:
- Drone telemetry (position, altitude, heading)
- Camera geometry (FOV, mount pitch/yaw)
- Optional terrain height for improved accuracy

Based on the flat_ground pinhole model from sentinel.geolocalizer.flat_ground
but optimized for per-frame ISR tracker use — no Track/VehicleState dependency,
just raw telemetry values and bbox coordinates.

Does NOT implement SLAM. Uses practical ISR approximations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class ProjectionReason:
    OK = "OK"
    NO_DRONE_POSE = "NO_DRONE_POSE"
    NAN_INPUT = "NAN_INPUT"
    INVALID_INTRINSICS = "INVALID_INTRINSICS"
    RAY_ABOVE_HORIZON = "RAY_ABOVE_HORIZON"
    ALTITUDE_TOO_LOW = "ALTITUDE_TOO_LOW"
    RAY_NEAR_PARALLEL = "RAY_NEAR_PARALLEL"


@dataclass
class DronePose:
    """Minimal drone state needed for projection."""
    lat: float = 0.0        # degrees
    lon: float = float("nan")  # degrees
    alt_m: float = 0.0       # relative altitude meters
    heading_deg: float = 0.0  # degrees clockwise from north
    groundspeed_mps: float = 0.0


@dataclass
class CameraParams:
    """Camera intrinsics and mount geometry."""
    width_px: int = 640
    height_px: int = 480
    h_fov_deg: float = 70.0
    v_fov_deg: float = 45.0
    pitch_deg: float = -45.0   # camera mount tilt (negative = down)
    yaw_offset_deg: float = 0.0


@dataclass
class WorldEstimate:
    """Result of projecting a pixel bbox to world coordinates."""
    north_m: float = 0.0       # meters from drone
    east_m: float = 0.0        # meters from drone
    lat: float = 0.0
    lon: float = float("nan")
    altitude_m: float = 0.0    # estimated target ground altitude
    ground_distance_m: float = 0.0  # horizontal distance from drone
    bearing_deg: float = 0.0
    uncertainty_m: float = 10.0
    valid: bool = False
    failure_reason: str = ""


# Approximate conversion factors
_M_PER_DEG_LAT = 111320.0


def _m_per_deg_lon(lat: float) -> float:
    return 111320.0 * math.cos(math.radians(lat))


def project_bbox_to_world(
    bbox: tuple[float, float, float, float],
    drone: DronePose,
    camera: CameraParams,
    terrain_height_m: float | None = None,
) -> WorldEstimate:
    """Project bbox bottom-center to ground plane.

    Uses bbox bottom-center (feet position) rather than center for
    more accurate ground contact estimation.

    Args:
        bbox: (x1, y1, x2, y2) in pixels
        drone: Current drone pose
        camera: Camera parameters
        terrain_height_m: Optional terrain height at target location.
            If None, uses drone altitude (flat-ground assumption).

    Returns:
        WorldEstimate with world coordinates and uncertainty.
    """
    x1, y1, x2, y2 = bbox

    # Validate drone pose
    for val in (drone.lat, drone.lon, drone.alt_m, drone.heading_deg):
        if not math.isfinite(val):
            return WorldEstimate(valid=False, failure_reason=ProjectionReason.NAN_INPUT)

    if camera.width_px <= 0 or camera.height_px <= 0:
        return WorldEstimate(valid=False, failure_reason=ProjectionReason.INVALID_INTRINSICS)

    # Use bottom-center of bbox (feet/ground contact point)
    cx_px = (x1 + x2) / 2.0
    cy_px = y2  # bottom of bbox = feet

    # Normalize pixel positions to [-0.5, 0.5]
    norm_x = (cx_px - camera.width_px / 2.0) / camera.width_px
    norm_y = (cy_px - camera.height_px / 2.0) / camera.height_px

    # Angular offsets from optical center
    horizontal_angle_deg = norm_x * camera.h_fov_deg
    vertical_offset_deg = norm_y * camera.v_fov_deg

    # Total pitch = mount pitch + vertical pixel offset
    pitch_total_deg = camera.pitch_deg + vertical_offset_deg

    if pitch_total_deg >= 0:
        return WorldEstimate(valid=False, failure_reason=ProjectionReason.RAY_ABOVE_HORIZON)

    # Altitude above ground
    ground_alt = terrain_height_m if terrain_height_m is not None else 0.0
    altitude_agl = drone.alt_m - ground_alt
    if altitude_agl <= 0:
        return WorldEstimate(valid=False, failure_reason=ProjectionReason.ALTITUDE_TOO_LOW)

    # Ground distance from tan(pitch) geometry
    pitch_rad = math.radians(abs(pitch_total_deg))
    tan_pitch = math.tan(pitch_rad)
    if tan_pitch < 1e-6:
        return WorldEstimate(valid=False, failure_reason=ProjectionReason.RAY_NEAR_PARALLEL)

    ground_distance_m = altitude_agl / tan_pitch

    # Bearing from heading + camera yaw + pixel horizontal offset
    bearing_deg = drone.heading_deg + camera.yaw_offset_deg + horizontal_angle_deg
    bearing_rad = math.radians(bearing_deg)

    north_m = ground_distance_m * math.cos(bearing_rad)
    east_m = ground_distance_m * math.sin(bearing_rad)

    # Convert to lat/lon
    dlat = north_m / _M_PER_DEG_LAT
    dlon = east_m / _m_per_deg_lon(drone.lat)
    target_lat = drone.lat + dlat
    target_lon = drone.lon + dlon

    # Uncertainty: grows with distance, off-center angle, and low pitch
    dist_factor = 1.0 + ground_distance_m / 50.0
    angle_factor = 1.0 + abs(horizontal_angle_deg) / max(camera.h_fov_deg, 1.0)
    pitch_factor = 1.0 / max(abs(pitch_total_deg) / 45.0, 0.3)
    uncertainty = 5.0 * dist_factor * angle_factor * pitch_factor

    return WorldEstimate(
        north_m=north_m,
        east_m=east_m,
        lat=target_lat,
        lon=target_lon,
        altitude_m=ground_alt,
        ground_distance_m=ground_distance_m,
        bearing_deg=bearing_deg,
        uncertainty_m=uncertainty,
        valid=True,
        failure_reason=ProjectionReason.OK,
    )


def estimate_world_velocity(
    prev_world: WorldEstimate | None,
    curr_world: WorldEstimate | None,
    dt_s: float,
) -> tuple[float, float, float]:
    """Estimate world-space velocity from two consecutive world estimates.

    Returns (speed_mps, heading_rad, uncertainty_mps).
    """
    if prev_world is None or curr_world is None or not prev_world.valid or not curr_world.valid:
        return 0.0, 0.0, 0.0
    if dt_s < 0.01:
        return 0.0, 0.0, 0.0

    dn = curr_world.north_m - prev_world.north_m
    de = curr_world.east_m - prev_world.east_m
    speed = math.sqrt(dn * dn + de * de) / dt_s
    heading = math.atan2(de, dn)
    uncertainty = (prev_world.uncertainty_m + curr_world.uncertainty_m) / (2.0 * dt_s)

    return speed, heading, uncertainty


def latlon_to_local(
    lat: float, lon: float,
    origin_lat: float, origin_lon: float,
) -> tuple[float, float]:
    """Convert lat/lon to local north/east meters from origin."""
    north_m = (lat - origin_lat) * _M_PER_DEG_LAT
    east_m = (lon - origin_lon) * _m_per_deg_lon(origin_lat)
    return north_m, east_m


def local_to_latlon(
    north_m: float, east_m: float,
    origin_lat: float, origin_lon: float,
) -> tuple[float, float]:
    """Convert local north/east meters to lat/lon."""
    lat = origin_lat + north_m / _M_PER_DEG_LAT
    lon = origin_lon + east_m / _m_per_deg_lon(origin_lat)
    return lat, lon
