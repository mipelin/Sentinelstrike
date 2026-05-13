"""Tests for SensorPose unified model."""

from sentinel.common.sensor_pose import (
    SensorPose,
    build_sensor_pose,
    sensor_pose_to_vehicle_state,
)
from sentinel.common.types import CameraModel, GeoPoint, VehicleState


def _vehicle(
    lat: float = 40.0,
    lon: float = -3.0,
    alt_m: float = 100.0,
    heading: float = 90.0,
    speed: float = 10.0,
) -> VehicleState:
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc="2025-01-01T00:00:00Z",
        position=GeoPoint(lat=lat, lon=lon, alt_m=alt_m),
        heading_deg=heading,
        groundspeed_mps=speed,
    )


def test_sensor_pose_validates_ranges():
    pose = SensorPose(
        vehicle_id="uav_001",
        frame_timestamp_utc="2025-01-01T00:00:00Z",
        heading_deg=180.0,
        position=GeoPoint(lat=40.0, lon=-3.0, alt_m=100.0),
    )
    assert pose.heading_deg == 180.0
    assert pose.position.lat == 40.0


def test_sensor_pose_rejects_bad_heading():
    import pytest
    with pytest.raises(Exception):
        SensorPose(
            vehicle_id="uav_001",
            frame_timestamp_utc="2025-01-01T00:00:00Z",
            heading_deg=400.0,
        )


def test_build_sensor_pose_preserves_fields():
    vs = _vehicle()
    cam = CameraModel()
    pose = build_sensor_pose(vs, "2025-01-01T00:00:01Z", cam, 1.5)
    assert pose.vehicle_id == "uav_001"
    assert pose.position.lat == 40.0
    assert pose.position.alt_m == 100.0
    assert pose.heading_deg == 90.0
    assert pose.groundspeed_mps == 10.0
    assert pose.camera is not None
    assert pose.capture_timestamp_monotonic_s == 1.5
    assert pose.altitude_above_ground_m == 100.0


def test_build_sensor_pose_no_camera():
    vs = _vehicle()
    pose = build_sensor_pose(vs, "2025-01-01T00:00:01Z")
    assert pose.camera is None


def test_roundtrip_vehicle_state():
    vs = _vehicle()
    pose = build_sensor_pose(vs, "2025-01-01T00:00:01Z")
    vs2 = sensor_pose_to_vehicle_state(pose)
    assert vs2.vehicle_id == vs.vehicle_id
    assert vs2.position.lat == vs.position.lat
    assert vs2.heading_deg == vs.heading_deg
    assert vs2.groundspeed_mps == vs.groundspeed_mps
    assert vs2.mode == vs.mode


def test_geolocalizer_accepts_sensor_pose():
    from sentinel.common.types import BoundingBox, Track
    from sentinel.geolocalizer.flat_ground import FlatGroundGeolocalizer

    geo = FlatGroundGeolocalizer(
        camera=CameraModel(pitch_deg=-45.0),
        assumed_ground_alt_m=0.0,
    )
    vs = _vehicle()
    pose = build_sensor_pose(vs, "2025-01-01T00:00:01Z", CameraModel(pitch_deg=-45.0))
    track = Track(
        track_id="trk_001",
        class_name="person",
        confidence=0.9,
        bbox_xyxy=BoundingBox(x1=300, y1=200, x2=340, y2=280),
        last_seen_utc="2025-01-01T00:00:01Z",
    )
    obs = geo.estimate_track_location_from_pose("mission_1", track, pose)
    assert obs is not None
    assert obs.estimated_location.lat != 0.0
    assert obs.track_id == "trk_001"


def test_backward_compat_vehicle_state():
    """Existing VehicleState path still works identically."""
    from sentinel.common.types import BoundingBox, Track
    from sentinel.geolocalizer.flat_ground import FlatGroundGeolocalizer

    geo = FlatGroundGeolocalizer(
        camera=CameraModel(pitch_deg=-45.0),
        assumed_ground_alt_m=0.0,
    )
    vs = _vehicle()
    track = Track(
        track_id="trk_001",
        class_name="person",
        confidence=0.9,
        bbox_xyxy=BoundingBox(x1=300, y1=200, x2=340, y2=280),
        last_seen_utc="2025-01-01T00:00:01Z",
    )
    obs = geo.estimate_track_location("mission_1", track, vs)
    assert obs is not None
    assert obs.track_id == "trk_001"
