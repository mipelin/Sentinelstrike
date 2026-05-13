"""Safety policy for MAVLink commands."""

from __future__ import annotations

from pydantic import BaseModel

from .commands import MavlinkCommand
from .telemetry import TelemetrySnapshot


class MavlinkSafetyConfig(BaseModel):
    allow_arm: bool = False
    allow_takeoff: bool = False
    allow_real_backend: bool = False
    max_takeoff_altitude_m: float = 30
    max_goto_distance_m: float = 1000
    require_simulation_mode: bool = True


_SAFE_COMMANDS = frozenset(
    {
        "CONNECT",
        "GET_TELEMETRY",
        "HOLD",
        "LAND",
        "RETURN_HOME",
        "DISARM",
    }
)


class SafetyPolicy:
    def __init__(self, config: MavlinkSafetyConfig, system_mode: str) -> None:
        self._config = config
        self._system_mode = system_mode

    def validate_command(
        self,
        command: MavlinkCommand,
        current_telemetry: TelemetrySnapshot | None = None,
    ) -> None:
        ct = command.command_type

        if ct in _SAFE_COMMANDS:
            return

        if self._config.require_simulation_mode and self._system_mode != "SIMULATION_MODE":
            raise ValueError(f"Command {ct} rejected: system is not in SIMULATION_MODE (current: {self._system_mode})")

        if ct == "ARM" and not self._config.allow_arm:
            raise ValueError("ARM is not allowed by safety policy")

        if ct == "TAKEOFF":
            if not self._config.allow_takeoff:
                raise ValueError("TAKEOFF is not allowed by safety policy")
            alt = command.payload.get("altitude_m")
            if alt is not None and float(alt) > self._config.max_takeoff_altitude_m:
                raise ValueError(f"TAKEOFF altitude {alt}m exceeds maximum {self._config.max_takeoff_altitude_m}m")

        if ct == "GOTO":
            if current_telemetry and current_telemetry.position:
                from sentinel.common.geo import haversine_distance_m

                lat = command.payload.get("lat")
                lon = command.payload.get("lon")
                if lat is not None and lon is not None:
                    from sentinel.common.types import GeoPoint

                    target = GeoPoint(lat=float(lat), lon=float(lon))
                    dist = haversine_distance_m(current_telemetry.position, target)
                    if dist > self._config.max_goto_distance_m:
                        raise ValueError(
                            f"GOTO distance {dist:.0f}m exceeds maximum {self._config.max_goto_distance_m}m"
                        )

        if ct == "UPLOAD_MISSION" or ct == "START_MISSION":
            pass  # allowed in simulation mode

    def check_real_backend_allowed(self) -> None:
        if not self._config.allow_real_backend:
            raise ValueError("Real backend is not allowed by safety policy")
