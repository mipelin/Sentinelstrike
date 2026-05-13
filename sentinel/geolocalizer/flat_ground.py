"""Flat-ground geolocalizer using pinhole camera model.

Assumptions:
- Flat terrain at assumed_ground_alt_m.
- Camera mounted on vehicle with known pitch/yaw offset.
- No terrain elevation model.
- No lens distortion.
- Heading is true north-referenced (clockwise from north).

Limitations:
- Accuracy degrades significantly when pitch is near horizontal.
- No correction for roll (assumed 0 for ground projection).
- Distance estimate grows quadratically with small pitch errors near horizon.
"""

from __future__ import annotations

import math

from loguru import logger

from sentinel.common.geo import offset_point
from sentinel.common.time import utc_now_iso
from sentinel.common.types import (
    CameraModel,
    GeoObservation,
    Track,
    VehicleState,
)
from sentinel.common.sensor_pose import SensorPose, sensor_pose_to_vehicle_state

from .camera import bbox_center_px
from .observation import make_observation_id


class FlatGroundGeolocalizer:
    def __init__(
        self,
        camera: CameraModel,
        assumed_ground_alt_m: float = 0.0,
        default_accuracy_estimate_m: float = 35.0,
    ) -> None:
        self._camera = camera
        self._assumed_ground_alt_m = assumed_ground_alt_m
        self._default_accuracy_estimate_m = default_accuracy_estimate_m

    def estimate_track_location(
        self,
        mission_id: str,
        track: Track,
        vehicle_state: VehicleState,
        timestamp_utc: str | None = None,
    ) -> GeoObservation | None:
        if track.bbox_xyxy is None:
            return None
        if vehicle_state.position is None:
            return None
        if vehicle_state.position.alt_m is None:
            return None

        ts = timestamp_utc or utc_now_iso()
        heading_deg = vehicle_state.heading_deg if vehicle_state.heading_deg is not None else 0.0

        # Pixel center of detection bbox
        cx_px, cy_px = bbox_center_px(track.bbox_xyxy)

        # Horizontal angle offset from optical center (degrees)
        norm_x = (cx_px - self._camera.width_px / 2.0) / self._camera.width_px
        horizontal_angle_deg = norm_x * self._camera.horizontal_fov_deg

        # Vertical angle offset from optical center (degrees, positive = down)
        norm_y = (cy_px - self._camera.height_px / 2.0) / self._camera.height_px
        vertical_offset_deg = norm_y * self._camera.vertical_fov_deg

        # Total pitch = camera mount pitch + vertical pixel offset
        # pitch_deg is negative for downward-looking, e.g. -45
        pitch_total_deg = self._camera.pitch_deg + vertical_offset_deg

        # Must be looking downward
        if pitch_total_deg >= 0:
            logger.debug("Pitch not downward enough: {:.1f} deg, skipping track {}", pitch_total_deg, track.track_id)
            return None

        altitude_above_ground = vehicle_state.position.alt_m - self._assumed_ground_alt_m
        if altitude_above_ground <= 0:
            return None

        pitch_total_rad = math.radians(abs(pitch_total_deg))
        tan_pitch = math.tan(pitch_total_rad)
        if tan_pitch < 1e-6:
            return None

        ground_distance_m = altitude_above_ground / tan_pitch

        bearing_deg = heading_deg + self._camera.yaw_offset_deg + horizontal_angle_deg

        bearing_rad = math.radians(bearing_deg)
        north_m = ground_distance_m * math.cos(bearing_rad)
        east_m = ground_distance_m * math.sin(bearing_rad)

        estimated_location = offset_point(
            origin=vehicle_state.position,
            north_m=north_m,
            east_m=east_m,
            alt_m=self._assumed_ground_alt_m,
        )

        # Accuracy degrades with off-center angle and near-horizon pitch
        accuracy = self._default_accuracy_estimate_m * (
            1.0 + abs(horizontal_angle_deg) / self._camera.horizontal_fov_deg
        )

        return GeoObservation(
            observation_id=make_observation_id(track.track_id, ts),
            mission_id=mission_id,
            track_id=track.track_id,
            class_name=track.class_name,
            timestamp_utc=ts,
            estimated_location=estimated_location,
            accuracy_estimate_m=round(accuracy, 1),
            method="flat_ground_pinhole_v1",
            confidence=track.confidence,
            source="geolocalizer",
            metadata={
                "ground_distance_m": round(ground_distance_m, 2),
                "bearing_deg": round(bearing_deg, 2),
                "horizontal_angle_deg": round(horizontal_angle_deg, 2),
                "vertical_angle_deg": round(vertical_offset_deg, 2),
                "altitude_above_ground_m": round(altitude_above_ground, 2),
            },
        )

    def estimate_track_location_from_pose(
        self,
        mission_id: str,
        track: Track,
        sensor_pose: SensorPose,
        timestamp_utc: str | None = None,
    ) -> GeoObservation | None:
        """Estimate track location using a SensorPose instead of VehicleState.

        Converts SensorPose to VehicleState internally and delegates to
        estimate_track_location. Purely additive, no breaking change.
        """
        vehicle_state = sensor_pose_to_vehicle_state(sensor_pose)
        return self.estimate_track_location(
            mission_id=mission_id,
            track=track,
            vehicle_state=vehicle_state,
            timestamp_utc=timestamp_utc or sensor_pose.frame_timestamp_utc,
        )
