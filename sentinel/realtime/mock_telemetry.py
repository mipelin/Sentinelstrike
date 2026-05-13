"""Mock telemetry provider with simulated UAV movement along waypoints."""

from __future__ import annotations

import math

from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, VehicleState


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _offset_point(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    R = 6371000.0
    d = dist_m / R
    b = math.radians(bearing_deg)
    lat_r = math.asin(
        math.sin(math.radians(lat)) * math.cos(d) +
        math.cos(math.radians(lat)) * math.sin(d) * math.cos(b)
    )
    lon_r = math.radians(lon) + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(math.radians(lat)),
        math.cos(d) - math.sin(math.radians(lat)) * math.sin(lat_r)
    )
    return math.degrees(lat_r), math.degrees(lon_r)


class MockMovementTelemetryProvider:
    """Simulates UAV movement along waypoints or in a small area."""

    def __init__(
        self,
        vehicle_id: str = "uav_001",
        waypoints: list[GeoPoint] | None = None,
        base_lat: float = 38.001,
        base_lon: float = -8.001,
        alt_m: float = 80.0,
        speed_mps: float = 10.0,
    ) -> None:
        self._vehicle_id = vehicle_id
        self._alt_m = alt_m
        self._speed_mps = speed_mps
        self._stale_count = 0

        if waypoints:
            self._waypoints = waypoints
        else:
            self._waypoints = [
                GeoPoint(lat=base_lat, lon=base_lon, alt_m=alt_m),
                GeoPoint(lat=base_lat + 0.001, lon=base_lon + 0.001, alt_m=alt_m),
                GeoPoint(lat=base_lat + 0.001, lon=base_lon - 0.001, alt_m=alt_m),
                GeoPoint(lat=base_lat, lon=base_lon, alt_m=alt_m),
            ]

        self._current_lat = self._waypoints[0].lat
        self._current_lon = self._waypoints[0].lon
        self._current_heading = 90.0
        self._current_speed = speed_mps
        self._current_battery = 95.0
        self._target_wp_idx = 1
        self._total_dist_m = 0.0

    @property
    def stale_count(self) -> int:
        return self._stale_count

    def get_vehicle_state(self) -> VehicleState:
        self._advance()
        self._current_battery = max(10.0, self._current_battery - 0.02)

        return VehicleState(
            vehicle_id=self._vehicle_id,
            timestamp_utc=utc_now_iso(),
            position=GeoPoint(lat=self._current_lat, lon=self._current_lon, alt_m=self._alt_m),
            heading_deg=self._current_heading,
            groundspeed_mps=self._current_speed,
            mode="AUTO",
            armed=True,
            battery_pct=round(self._current_battery, 1),
        )

    def _advance(self) -> None:
        if len(self._waypoints) < 2:
            return

        # Distance per tick at ~5 Hz
        step_m = self._speed_mps * 0.2  # ~2m per tick at 5Hz
        self._total_dist_m += step_m

        target = self._waypoints[self._target_wp_idx % len(self._waypoints)]
        dist = _haversine_m(self._current_lat, self._current_lon, target.lat, target.lon)

        if dist < step_m * 1.5:
            self._target_wp_idx = (self._target_wp_idx + 1) % len(self._waypoints)
            target = self._waypoints[self._target_wp_idx]
            dist = _haversine_m(self._current_lat, self._current_lon, target.lat, target.lon)

        if dist < 0.1:
            return

        heading = _bearing_deg(self._current_lat, self._current_lon, target.lat, target.lon)
        self._current_heading = heading

        move_m = min(step_m, dist)
        new_lat, new_lon = _offset_point(self._current_lat, self._current_lon, heading, move_m)
        self._current_lat = new_lat
        self._current_lon = new_lon
