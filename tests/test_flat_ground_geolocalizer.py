"""Tests for FlatGroundGeolocalizer."""

from sentinel.common.time import utc_now_iso
from sentinel.common.types import (
    BoundingBox,
    CameraModel,
    GeoPoint,
    Track,
    VehicleState,
)
from sentinel.geolocalizer.flat_ground import FlatGroundGeolocalizer


def _make_track(track_id="trk_000001", class_name="person", conf=0.9, x1=100, y1=50, x2=300, y2=250):
    return Track(
        track_id=track_id,
        class_name=class_name,
        confidence=conf,
        bbox_xyxy=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        last_seen_utc=utc_now_iso(),
    )


def _make_vehicle(lat=38.0, lon=-8.0, alt=80.0, heading=90.0):
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=lat, lon=lon, alt_m=alt),
        heading_deg=heading,
        groundspeed_mps=10.0,
        mode="SIMULATED",
        armed=False,
        battery_pct=90.0,
    )


def test_track_with_bbox_produces_observation():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    obs = geo.estimate_track_location(
        mission_id="m1",
        track=_make_track(),
        vehicle_state=_make_vehicle(),
    )
    assert obs is not None
    assert obs.estimated_location is not None
    assert obs.method == "flat_ground_pinhole_v1"
    assert obs.accuracy_estimate_m > 0


def test_without_bbox_returns_none():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    track = _make_track()
    track = track.model_copy(update={"bbox_xyxy": None})
    obs = geo.estimate_track_location("m1", track, _make_vehicle())
    assert obs is None


def test_without_position_returns_none():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    veh = VehicleState(vehicle_id="uav_001", timestamp_utc=utc_now_iso())
    obs = geo.estimate_track_location("m1", _make_track(), veh)
    assert obs is None


def test_without_altitude_returns_none():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    veh = VehicleState(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=38.0, lon=-8.0),
    )
    obs = geo.estimate_track_location("m1", _make_track(), veh)
    assert obs is None


def test_estimated_location_differs_from_vehicle():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    veh = _make_vehicle()
    obs = geo.estimate_track_location("m1", _make_track(), veh)
    assert obs is not None
    assert obs.estimated_location.lat != veh.position.lat or obs.estimated_location.lon != veh.position.lon


def test_metadata_includes_ground_distance_and_bearing():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    obs = geo.estimate_track_location("m1", _make_track(), _make_vehicle())
    assert obs is not None
    assert "ground_distance_m" in obs.metadata
    assert "bearing_deg" in obs.metadata
    assert obs.metadata["ground_distance_m"] > 0


def test_confidence_propagated_from_track():
    cam = CameraModel(pitch_deg=-45.0)
    geo = FlatGroundGeolocalizer(camera=cam)
    track = _make_track(conf=0.72)
    obs = geo.estimate_track_location("m1", track, _make_vehicle())
    assert obs is not None
    assert obs.confidence == 0.72
