"""MAVLink command types and result model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class CommandType(StrEnum):
    CONNECT = "CONNECT"
    ARM = "ARM"
    DISARM = "DISARM"
    TAKEOFF = "TAKEOFF"
    LAND = "LAND"
    RETURN_HOME = "RETURN_HOME"
    HOLD = "HOLD"
    GOTO = "GOTO"
    UPLOAD_MISSION = "UPLOAD_MISSION"
    START_MISSION = "START_MISSION"
    GET_TELEMETRY = "GET_TELEMETRY"
    OFFBOARD_START = "OFFBOARD_START"
    OFFBOARD_STOP = "OFFBOARD_STOP"
    OFFBOARD_SET_VELOCITY = "OFFBOARD_SET_VELOCITY"


class MavlinkCommand(BaseModel):
    command_id: str
    mission_id: str
    command_type: str
    timestamp_utc: str
    payload: dict = {}


class CommandResult(BaseModel):
    command_id: str
    success: bool
    message: str
    timestamp_utc: str
    payload: dict = {}
