"""Telemetry snapshot model."""

from __future__ import annotations

import math

from loguru import logger
from pydantic import BaseModel

from sentinel.common.types import GeoPoint, VehicleState


def normalize_battery_pct(value: float | None) -> float | None:
    """Normalize battery percentage from various MAVSDK/PX4 scalings.

    MAVSDK reports ``remaining_percent`` in 0..1, so ``* 100`` should give
    0..100.  However, some PX4 SITL builds report it already scaled to
    0..100, yielding values like 8900.0 after multiplication.  This helper
    handles all known cases and clamps the result to [0, 100].

    Returns None for missing/invalid values.  Rounds to 2 decimals.
    """
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if value < 0:
        return None
    if value <= 1.0:
        return round(value * 100, 2)
    if value <= 100.0:
        return round(value, 2)
    # value > 100 — likely double-scaled (e.g. MAVSDK 0..1 * 100 on a
    # value that was already 0..100, giving 8900.0).
    if value <= 10_000.0:
        converted = value / 100.0
        if converted <= 100.0:
            return round(converted, 2)
    # Truly out of range — clamp with a warning rather than crashing.
    logger.warning("battery_pct={} is out of range, clamping to 100.0", value)
    return 100.0


class TelemetrySnapshot(BaseModel):
    vehicle_id: str
    timestamp_utc: str
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    position: GeoPoint | None = None
    heading_deg: float | None = None
    groundspeed_mps: float | None = None
    battery_pct: float | None = None
    last_command: str | None = None

    def to_vehicle_state(self) -> VehicleState:
        return VehicleState(
            vehicle_id=self.vehicle_id,
            timestamp_utc=self.timestamp_utc,
            position=self.position,
            heading_deg=self.heading_deg,
            groundspeed_mps=self.groundspeed_mps,
            mode=self.mode,
            armed=self.armed,
            battery_pct=normalize_battery_pct(self.battery_pct),
        )
