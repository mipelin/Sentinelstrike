"""Telemetry providers for the real-time loop."""

from __future__ import annotations

import time

from loguru import logger

from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, VehicleState


class StaticTelemetryProvider:
    """Returns a fixed simulated VehicleState."""

    def __init__(
        self,
        vehicle_id: str = "uav_001",
        lat: float = 38.001,
        lon: float = -8.001,
        alt_m: float = 80.0,
        heading_deg: float = 90.0,
    ) -> None:
        self._state = VehicleState(
            vehicle_id=vehicle_id,
            timestamp_utc=utc_now_iso(),
            position=GeoPoint(lat=lat, lon=lon, alt_m=alt_m),
            heading_deg=heading_deg,
            groundspeed_mps=10.0,
            mode="SIMULATED",
            armed=False,
            battery_pct=90.0,
        )

    def get_vehicle_state(self) -> VehicleState:
        return self._state.model_copy(update={"timestamp_utc": utc_now_iso()})


class MavlinkTelemetryProvider:
    """Reads live telemetry from a MavlinkBridge, with stale fallback and age tracking."""

    def __init__(
        self,
        mavlink_bridge: object,
        vehicle_id: str = "uav_001",
        telemetry_timeout_s: float = 3.0,
    ) -> None:
        self._bridge = mavlink_bridge
        self._vehicle_id = vehicle_id
        self._last_valid: VehicleState | None = None
        self._stale_count = 0
        self._last_success_monotonic: float = 0.0
        self._last_age_ms: float = 0.0
        self._link_ok: bool = False
        self._timeout_s = telemetry_timeout_s

    @property
    def stale_count(self) -> int:
        return self._stale_count

    @property
    def telemetry_age_ms(self) -> float:
        return self._last_age_ms

    @property
    def link_ok(self) -> bool:
        return self._link_ok

    def get_vehicle_state(self) -> VehicleState | None:
        try:
            t0 = time.monotonic()
            snap = self._bridge.get_telemetry()
            elapsed = (time.monotonic() - t0) * 1000.0
            if elapsed > self._timeout_s * 1000:
                logger.warning("Telemetry read took {:.0f}ms (timeout {:.0f}s)", elapsed, self._timeout_s)

            state = snap.to_vehicle_state()
            self._last_valid = state
            self._last_success_monotonic = time.monotonic()
            self._last_age_ms = 0.0
            self._link_ok = True
            self._stale_count = 0
            return state
        except Exception as exc:
            logger.warning("Telemetry read failed: {}", exc)
            self._stale_count += 1
            self._link_ok = False
            if self._last_valid is not None and self._last_success_monotonic > 0:
                self._last_age_ms = (time.monotonic() - self._last_success_monotonic) * 1000.0
                return self._last_valid.model_copy(
                    update={
                        "timestamp_utc": utc_now_iso(),
                    }
                )
            return None
