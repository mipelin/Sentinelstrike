"""MAVLink bridge — orchestrates backend calls, safety, and event publishing."""

from __future__ import annotations

import uuid

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionPlan

from .backends import MavlinkBackend
from .commands import CommandResult, CommandType, MavlinkCommand
from .safety import SafetyPolicy
from .telemetry import TelemetrySnapshot


class MavlinkBridge:
    def __init__(
        self,
        mission_id: str,
        vehicle_id: str,
        backend: MavlinkBackend,
        event_bus: EventBus | None = None,
        safety_policy: SafetyPolicy | None = None,
    ) -> None:
        self._mission_id = mission_id
        self._vehicle_id = vehicle_id
        self._backend = backend
        self._event_bus = event_bus
        self._safety = safety_policy

    def _make_command(self, command_type: str | CommandType, **payload: object) -> MavlinkCommand:
        return MavlinkCommand(
            command_id=uuid.uuid4().hex[:12],
            mission_id=self._mission_id,
            command_type=str(command_type),
            timestamp_utc=utc_now_iso(),
            payload=payload,
        )

    def _publish(self, event_type: str, **payload: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="mavlink_bridge",
                    event_type=event_type,
                    payload=payload,
                )
            )

    def execute(self, command: MavlinkCommand) -> CommandResult:
        self._publish(
            "mavlink_command_requested",
            command_type=command.command_type,
            command_id=command.command_id,
        )
        if self._safety is not None:
            try:
                self._safety.validate_command(command)
            except ValueError as exc:
                result = CommandResult(
                    command_id=command.command_id,
                    success=False,
                    message=str(exc),
                    timestamp_utc=utc_now_iso(),
                )
                self._publish(
                    "mavlink_command_failed",
                    command_type=command.command_type,
                    command_id=command.command_id,
                    message=str(exc),
                )
                return result

        try:
            result = self._dispatch(command)
        except Exception as exc:
            logger.exception("MAVLink backend error for {}", command.command_type)
            result = CommandResult(
                command_id=command.command_id,
                success=False,
                message=str(exc),
                timestamp_utc=utc_now_iso(),
            )
            self._publish(
                "mavlink_command_failed",
                command_type=command.command_type,
                command_id=command.command_id,
                message=str(exc),
            )
            return result

        event_name = "mavlink_command_succeeded" if result.success else "mavlink_command_failed"
        self._publish(
            event_name,
            command_type=command.command_type,
            command_id=command.command_id,
            success=result.success,
            message=result.message,
        )
        return result

    def _dispatch(self, command: MavlinkCommand) -> CommandResult:
        ct = command.command_type
        if ct == CommandType.CONNECT:
            return self._backend.connect()
        if ct == CommandType.ARM:
            return self._backend.arm()
        if ct == CommandType.DISARM:
            return self._backend.disarm()
        if ct == CommandType.TAKEOFF:
            return self._backend.takeoff(altitude_m=float(command.payload.get("altitude_m", 0)))
        if ct == CommandType.LAND:
            return self._backend.land()
        if ct == CommandType.RETURN_HOME:
            return self._backend.return_home()
        if ct == CommandType.HOLD:
            return self._backend.hold()
        if ct == CommandType.GOTO:
            p = command.payload
            return self._backend.goto(GeoPoint(lat=float(p["lat"]), lon=float(p["lon"]), alt_m=p.get("alt_m")))
        if ct == CommandType.UPLOAD_MISSION:
            raise ValueError("Use bridge.upload_mission() for mission uploads")
        if ct == CommandType.START_MISSION:
            return self._backend.start_mission()
        if ct == CommandType.GET_TELEMETRY:
            raise ValueError("Use bridge.get_telemetry() for telemetry queries")
        raise ValueError(f"Unknown command type: {ct}")

    # --- convenience methods ---

    def connect(self) -> CommandResult:
        cmd = self._make_command(CommandType.CONNECT)
        return self.execute(cmd)

    def arm(self) -> CommandResult:
        cmd = self._make_command(CommandType.ARM)
        return self.execute(cmd)

    def disarm(self) -> CommandResult:
        cmd = self._make_command(CommandType.DISARM)
        return self.execute(cmd)

    def takeoff(self, altitude_m: float) -> CommandResult:
        cmd = self._make_command(CommandType.TAKEOFF, altitude_m=altitude_m)
        return self.execute(cmd)

    def land(self) -> CommandResult:
        cmd = self._make_command(CommandType.LAND)
        return self.execute(cmd)

    def return_home(self) -> CommandResult:
        cmd = self._make_command(CommandType.RETURN_HOME)
        return self.execute(cmd)

    def hold(self) -> CommandResult:
        cmd = self._make_command(CommandType.HOLD)
        return self.execute(cmd)

    def goto(self, point: GeoPoint) -> CommandResult:
        cmd = self._make_command(CommandType.GOTO, lat=point.lat, lon=point.lon, alt_m=point.alt_m)
        return self.execute(cmd)

    def upload_mission(self, plan: MissionPlan) -> CommandResult:
        cmd = self._make_command(CommandType.UPLOAD_MISSION)
        self._publish(
            "mavlink_command_requested",
            command_type=CommandType.UPLOAD_MISSION,
            command_id=cmd.command_id,
        )
        if self._safety is not None:
            try:
                self._safety.validate_command(cmd)
            except ValueError as exc:
                result = CommandResult(
                    command_id=cmd.command_id,
                    success=False,
                    message=str(exc),
                    timestamp_utc=utc_now_iso(),
                )
                self._publish(
                    "mavlink_command_failed",
                    command_type=CommandType.UPLOAD_MISSION,
                    command_id=cmd.command_id,
                    message=str(exc),
                )
                return result
        try:
            result = self._backend.upload_mission(plan)
        except Exception as exc:
            result = CommandResult(
                command_id=cmd.command_id,
                success=False,
                message=str(exc),
                timestamp_utc=utc_now_iso(),
            )
        event_name = "mavlink_command_succeeded" if result.success else "mavlink_command_failed"
        self._publish(
            event_name,
            command_type=CommandType.UPLOAD_MISSION,
            command_id=cmd.command_id,
            success=result.success,
            message=result.message,
        )
        return result

    def start_mission(self) -> CommandResult:
        cmd = self._make_command(CommandType.START_MISSION)
        return self.execute(cmd)

    def get_telemetry(self) -> TelemetrySnapshot:
        telemetry = self._backend.get_telemetry()
        self._publish(
            "mavlink_telemetry_received",
            vehicle_id=telemetry.vehicle_id,
            mode=telemetry.mode,
            armed=telemetry.armed,
            connected=telemetry.connected,
        )
        return telemetry

    def close(self) -> None:
        self._backend.close()
