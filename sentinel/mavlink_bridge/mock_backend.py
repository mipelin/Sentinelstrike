"""Mock MAVLink backend for simulation and testing."""

from __future__ import annotations

import uuid

from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionPlan

from .commands import CommandResult
from .telemetry import TelemetrySnapshot


def _result(success: bool, message: str, **payload: object) -> CommandResult:
    return CommandResult(
        command_id=uuid.uuid4().hex[:12],
        success=success,
        message=message,
        timestamp_utc=utc_now_iso(),
        payload=payload,
    )


class MockMavlinkBackend:
    def __init__(self, vehicle_id: str = "uav_001") -> None:
        self._vehicle_id = vehicle_id
        self.connected: bool = False
        self.armed: bool = False
        self.mode: str = "UNKNOWN"
        self.current_position: GeoPoint | None = None
        self.battery_pct: float = 100.0
        self.last_command: str | None = None
        self._uploaded_mission: MissionPlan | None = None

    def connect(self) -> CommandResult:
        self.connected = True
        self.last_command = "CONNECT"
        return _result(True, "Mock connected")

    def arm(self) -> CommandResult:
        if not self.connected:
            return _result(False, "Not connected")
        self.armed = True
        self.last_command = "ARM"
        return _result(True, "Mock armed")

    def disarm(self) -> CommandResult:
        self.armed = False
        self.last_command = "DISARM"
        return _result(True, "Mock disarmed")

    def takeoff(self, altitude_m: float) -> CommandResult:
        if not self.connected:
            return _result(False, "Not connected")
        if not self.armed:
            return _result(False, "Not armed")
        self.mode = "TAKEOFF"
        self.last_command = "TAKEOFF"
        if self.current_position is not None:
            self.current_position = GeoPoint(
                lat=self.current_position.lat,
                lon=self.current_position.lon,
                alt_m=altitude_m,
            )
        return _result(True, "Mock takeoff", altitude_m=altitude_m)

    def land(self) -> CommandResult:
        self.mode = "LAND"
        self.armed = False
        self.last_command = "LAND"
        if self.current_position is not None:
            self.current_position = GeoPoint(
                lat=self.current_position.lat,
                lon=self.current_position.lon,
                alt_m=0,
            )
        return _result(True, "Mock landed")

    def return_home(self) -> CommandResult:
        self.mode = "RETURN_HOME"
        self.last_command = "RETURN_HOME"
        return _result(True, "Mock return home")

    def hold(self) -> CommandResult:
        self.mode = "HOLD"
        self.last_command = "HOLD"
        return _result(True, "Mock hold")

    def goto(self, point: GeoPoint) -> CommandResult:
        if not self.connected:
            return _result(False, "Not connected")
        self.current_position = point
        self.mode = "GOTO"
        self.last_command = "GOTO"
        return _result(True, "Mock goto", lat=point.lat, lon=point.lon)

    def upload_mission(self, plan: MissionPlan) -> CommandResult:
        if not self.connected:
            return _result(False, "Not connected")
        self._uploaded_mission = plan
        self.mode = "MISSION_UPLOADED"
        self.last_command = "UPLOAD_MISSION"
        return _result(True, "Mock mission uploaded", waypoints=len(plan.waypoints))

    def start_mission(self) -> CommandResult:
        if not self.connected:
            return _result(False, "Not connected")
        if self._uploaded_mission is None:
            return _result(False, "No mission uploaded")
        self.mode = "MISSION"
        self.last_command = "START_MISSION"
        if self._uploaded_mission.waypoints:
            self.current_position = self._uploaded_mission.waypoints[-1]
        return _result(True, "Mock mission started")

    def get_telemetry(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            vehicle_id=self._vehicle_id,
            timestamp_utc=utc_now_iso(),
            connected=self.connected,
            armed=self.armed,
            mode=self.mode,
            position=self.current_position,
            heading_deg=0.0,
            groundspeed_mps=0.0,
            battery_pct=self.battery_pct,
            last_command=self.last_command,
        )

    def close(self) -> None:
        self.connected = False
