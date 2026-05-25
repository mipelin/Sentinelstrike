"""Tests for world-space projection pipeline.

Validates project_bbox_to_world(), estimate_world_velocity(), lat/lon conversions,
ProjectionReason enum, NaN guards, and camera resolution handling.
"""

from __future__ import annotations

import math

import pytest

from sentinel.tracker.world_projection import (
    CameraParams,
    DronePose,
    ProjectionReason,
    WorldEstimate,
    estimate_world_velocity,
    latlon_to_local,
    local_to_latlon,
    project_bbox_to_world,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZURICH_LAT = 47.397971
_ZURICH_LON = 8.546164


def _drone_at(alt=50.0, heading=0.0, lat=_ZURICH_LAT, lon=_ZURICH_LON):
    return DronePose(lat=lat, lon=lon, alt_m=alt, heading_deg=heading)


def _cam_640x480():
    return CameraParams(width_px=640, height_px=480, h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0)


def _centered_bbox_640():
    """Bbox at center of 640x480 frame."""
    return (270, 190, 370, 290)


# ---------------------------------------------------------------------------
# ProjectionReason enum
# ---------------------------------------------------------------------------


class TestProjectionReason:
    def test_reasons_are_strings(self):
        assert isinstance(ProjectionReason.OK, str)
        assert isinstance(ProjectionReason.NAN_INPUT, str)
        assert isinstance(ProjectionReason.RAY_ABOVE_HORIZON, str)

    def test_ok_value(self):
        assert ProjectionReason.OK == "OK"

    def test_all_reasons_unique(self):
        reasons = [
            ProjectionReason.OK,
            ProjectionReason.NO_DRONE_POSE,
            ProjectionReason.NAN_INPUT,
            ProjectionReason.INVALID_INTRINSICS,
            ProjectionReason.RAY_ABOVE_HORIZON,
            ProjectionReason.ALTITUDE_TOO_LOW,
            ProjectionReason.RAY_NEAR_PARALLEL,
        ]
        assert len(reasons) == len(set(reasons))


# ---------------------------------------------------------------------------
# Valid projection
# ---------------------------------------------------------------------------


class TestValidProjection:
    def test_centered_target_valid(self):
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), _cam_640x480(),
        )
        assert est.valid
        assert est.failure_reason == ProjectionReason.OK

    def test_centered_target_ground_distance(self):
        alt = 50.0
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(alt), _cam_640x480(),
        )
        assert est.valid
        # pitch_deg=-45, cy_px=290 → vertical_offset = (290-240)/480 * 45 = 4.69°
        # pitch_total = -45 + 4.69 = -40.31°, ground_dist = alt / tan(40.31°)
        pitch_total = -45.0 + ((290 - 240) / 480.0) * 45.0
        expected_dist = alt / math.tan(math.radians(abs(pitch_total)))
        assert abs(est.ground_distance_m - expected_dist) < 1.0

    def test_centered_target_near_drone_position(self):
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), _cam_640x480(),
        )
        assert est.valid
        # Lat/lon should be near drone position
        assert abs(est.lat - _ZURICH_LAT) < 0.01
        assert abs(est.lon - _ZURICH_LON) < 0.01

    def test_uncertainty_positive(self):
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), _cam_640x480(),
        )
        assert est.valid
        assert est.uncertainty_m > 0


# ---------------------------------------------------------------------------
# Failure reasons
# ---------------------------------------------------------------------------


class TestFailureReasons:
    def test_nan_lon_returns_nan_input(self):
        drone = DronePose(lat=47.0, lon=float("nan"), alt_m=50.0, heading_deg=0.0)
        est = project_bbox_to_world(_centered_bbox_640(), drone, _cam_640x480())
        assert not est.valid
        assert est.failure_reason == ProjectionReason.NAN_INPUT

    def test_nan_alt_returns_nan_input(self):
        drone = DronePose(lat=47.0, lon=8.0, alt_m=float("nan"), heading_deg=0.0)
        est = project_bbox_to_world(_centered_bbox_640(), drone, _cam_640x480())
        assert not est.valid
        assert est.failure_reason == ProjectionReason.NAN_INPUT

    def test_inf_heading_returns_nan_input(self):
        drone = DronePose(lat=47.0, lon=8.0, alt_m=50.0, heading_deg=float("inf"))
        est = project_bbox_to_world(_centered_bbox_640(), drone, _cam_640x480())
        assert not est.valid
        assert est.failure_reason == ProjectionReason.NAN_INPUT

    def test_zero_width_returns_invalid_intrinsics(self):
        cam = CameraParams(width_px=0, height_px=480)
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), cam,
        )
        assert not est.valid
        assert est.failure_reason == ProjectionReason.INVALID_INTRINSICS

    def test_horizon_rejection(self):
        # Camera looking horizontal (pitch=0) → ray above horizon
        cam = CameraParams(width_px=640, height_px=480, pitch_deg=0.0)
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), cam,
        )
        assert not est.valid
        assert est.failure_reason == ProjectionReason.RAY_ABOVE_HORIZON

    def test_camera_looking_up_rejection(self):
        # Camera looking up (pitch > 0)
        cam = CameraParams(width_px=640, height_px=480, pitch_deg=10.0)
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), cam,
        )
        assert not est.valid
        assert est.failure_reason == ProjectionReason.RAY_ABOVE_HORIZON

    def test_zero_altitude_returns_too_low(self):
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(alt=0.0), _cam_640x480(),
        )
        assert not est.valid
        assert est.failure_reason == ProjectionReason.ALTITUDE_TOO_LOW

    def test_negative_altitude_returns_too_low(self):
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(alt=-5.0), _cam_640x480(),
        )
        assert not est.valid
        assert est.failure_reason == ProjectionReason.ALTITUDE_TOO_LOW

    def test_default_world_estimate_no_reason(self):
        est = WorldEstimate()
        assert not est.valid
        assert est.failure_reason == ""


# ---------------------------------------------------------------------------
# Camera resolution
# ---------------------------------------------------------------------------


class TestCameraResolution:
    def test_mismatched_resolution_wrong_offset(self):
        """Using 640-wide params on a 1280-wide frame gives wrong normalization."""
        # Bbox at center of 1280x720 frame
        bbox_1280 = (540, 310, 740, 410)
        cam_640 = CameraParams(width_px=640, height_px=480, h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0)

        est = project_bbox_to_world(bbox_1280, _drone_at(50.0), cam_640)
        assert est.valid
        # cx_px = 640, normalized as (640 - 320) / 640 = 0.5 → horizontal_angle = 35°
        # This is WRONG — target is centered but projection says 35° off-center
        assert est.bearing_deg != pytest.approx(0.0, abs=1.0)

    def test_correct_resolution_centered(self):
        """Using correct resolution gives near-zero offset for centered target."""
        bbox_1280 = (540, 310, 740, 410)
        cam_1280 = CameraParams(width_px=1280, height_px=720, h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0)

        est = project_bbox_to_world(bbox_1280, _drone_at(50.0), cam_1280)
        assert est.valid
        # cx_px = 640, normalized as (640 - 640) / 1280 = 0.0 → centered
        assert abs(est.bearing_deg - _drone_at(50.0).heading_deg) < 1.0

    def test_different_resolution_same_physical_target(self):
        """Same relative target position should produce similar ground_distance
        when params match frame size."""
        # 640x480: center horizontally, bottom at 75% of frame height
        # bbox: x center=320, y bottom=360 (480*0.75)
        est_640 = project_bbox_to_world(
            (270, 300, 370, 360), _drone_at(50.0),
            CameraParams(width_px=640, height_px=480, h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0),
        )
        # 1280x720: same relative position — center horizontal, bottom at 75%
        # bbox: x center=640, y bottom=540 (720*0.75)
        est_1280 = project_bbox_to_world(
            (540, 460, 740, 540), _drone_at(50.0),
            CameraParams(width_px=1280, height_px=720, h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0),
        )
        assert est_640.valid and est_1280.valid
        assert abs(est_640.ground_distance_m - est_1280.ground_distance_m) < 3.0


# ---------------------------------------------------------------------------
# Bearing / ENU
# ---------------------------------------------------------------------------


class TestBearing:
    def test_heading_north_goes_north(self):
        bbox = (270, 380, 370, 480)  # bottom-center of frame
        est = project_bbox_to_world(bbox, _drone_at(50.0, heading=0.0), _cam_640x480())
        assert est.valid
        assert est.north_m > 0
        assert abs(est.east_m) < 1.0

    def test_heading_east_goes_east(self):
        bbox = (270, 380, 370, 480)
        est = project_bbox_to_world(bbox, _drone_at(50.0, heading=90.0), _cam_640x480())
        assert est.valid
        assert abs(est.north_m) < 1.0
        assert est.east_m > 0

    def test_heading_south_goes_south(self):
        bbox = (270, 380, 370, 480)
        est = project_bbox_to_world(bbox, _drone_at(50.0, heading=180.0), _cam_640x480())
        assert est.valid
        assert est.north_m < 0

    def test_off_center_target_affects_bearing(self):
        # Target on right side of frame → bearing offset right
        est_left = project_bbox_to_world((50, 190, 150, 290), _drone_at(50.0, heading=0.0), _cam_640x480())
        est_right = project_bbox_to_world((490, 190, 590, 290), _drone_at(50.0, heading=0.0), _cam_640x480())
        assert est_left.valid and est_right.valid
        assert est_left.east_m < est_right.east_m


# ---------------------------------------------------------------------------
# Lat/lon conversions
# ---------------------------------------------------------------------------


class TestLatLonConversions:
    def test_round_trip(self):
        north_m, east_m = 100.0, 200.0
        lat, lon = local_to_latlon(north_m, east_m, _ZURICH_LAT, _ZURICH_LON)
        rt_north, rt_east = latlon_to_local(lat, lon, _ZURICH_LAT, _ZURICH_LON)
        assert abs(rt_north - north_m) < 0.1
        assert abs(rt_east - east_m) < 0.1

    def test_zero_offset_identity(self):
        lat, lon = local_to_latlon(0.0, 0.0, _ZURICH_LAT, _ZURICH_LON)
        assert lat == _ZURICH_LAT
        assert lon == _ZURICH_LON

    def test_north_increases_lat(self):
        lat, lon = local_to_latlon(100.0, 0.0, _ZURICH_LAT, _ZURICH_LON)
        assert lat > _ZURICH_LAT
        assert lon == pytest.approx(_ZURICH_LON, abs=1e-10)

    def test_east_increases_lon(self):
        lat, lon = local_to_latlon(0.0, 100.0, _ZURICH_LAT, _ZURICH_LON)
        assert lon > _ZURICH_LON
        assert lat == pytest.approx(_ZURICH_LAT, abs=1e-10)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_uncertainty_grows_with_altitude(self):
        est_low = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(30.0), _cam_640x480(),
        )
        est_high = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(100.0), _cam_640x480(),
        )
        assert est_low.valid and est_high.valid
        assert est_high.uncertainty_m > est_low.uncertainty_m

    def test_off_center_higher_uncertainty(self):
        est_center = project_bbox_to_world(
            (270, 190, 370, 290), _drone_at(50.0), _cam_640x480(),
        )
        est_edge = project_bbox_to_world(
            (10, 190, 110, 290), _drone_at(50.0), _cam_640x480(),
        )
        assert est_center.valid and est_edge.valid
        assert est_edge.uncertainty_m > est_center.uncertainty_m


# ---------------------------------------------------------------------------
# estimate_world_velocity
# ---------------------------------------------------------------------------


class TestWorldVelocity:
    def test_stationary_zero_velocity(self):
        est = project_bbox_to_world(
            _centered_bbox_640(), _drone_at(50.0), _cam_640x480(),
        )
        speed, heading, unc = estimate_world_velocity(est, est, 1.0)
        assert speed == pytest.approx(0.0, abs=0.01)

    def test_one_mps_northward(self):
        prev = WorldEstimate(valid=True, north_m=0.0, east_m=0.0, uncertainty_m=5.0)
        curr = WorldEstimate(valid=True, north_m=1.0, east_m=0.0, uncertainty_m=5.0)
        speed, heading, unc = estimate_world_velocity(prev, curr, 1.0)
        assert speed == pytest.approx(1.0, abs=0.01)
        assert heading == pytest.approx(0.0, abs=0.01)

    def test_none_inputs_return_zero(self):
        speed, heading, unc = estimate_world_velocity(None, None, 1.0)
        assert speed == 0.0

    def test_invalid_inputs_return_zero(self):
        prev = WorldEstimate(valid=False)
        curr = WorldEstimate(valid=True, north_m=1.0, east_m=0.0, uncertainty_m=5.0)
        speed, heading, unc = estimate_world_velocity(prev, curr, 1.0)
        assert speed == 0.0

    def test_zero_dt_returns_zero(self):
        prev = WorldEstimate(valid=True, north_m=0.0, east_m=0.0, uncertainty_m=5.0)
        curr = WorldEstimate(valid=True, north_m=1.0, east_m=0.0, uncertainty_m=5.0)
        speed, heading, unc = estimate_world_velocity(prev, curr, 0.001)
        assert speed == 0.0
