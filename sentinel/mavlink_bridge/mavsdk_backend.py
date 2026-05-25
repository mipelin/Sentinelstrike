"""MAVSDK backend for PX4 SITL integration.

Requires: pip install -e .[mavlink]
All MAVSDK calls are async internally.  This wrapper uses a **persistent**
event loop (``AsyncLoopThread``) so that gRPC channels and coroutines remain
valid across calls — unlike ``asyncio.run()`` which destroys the loop after
each invocation.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from loguru import logger

from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionPlan

from .async_loop import AsyncLoopThread
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


# ---------------------------------------------------------------------------
# MissionItem value helper — pure function, testable without MAVSDK
# ---------------------------------------------------------------------------


def _mission_item_values_from_waypoint(
    wp_lat: float,
    wp_lon: float,
    wp_alt_m: float | None,
    default_alt: float,
    speed: float,
) -> dict[str, Any]:
    """Return a dict of positional args for ``MissionItem(...)``.

    Extracted as a pure function so we can test the mapping without
    requiring the ``mavsdk`` package.
    """
    return {
        "latitude_deg": wp_lat,
        "longitude_deg": wp_lon,
        "relative_altitude_m": wp_alt_m if wp_alt_m is not None else default_alt,
        "speed_m_s": speed,
        "is_fly_through": True,
        "gimbal_pitch_deg": float("nan"),
        "gimbal_yaw_deg": float("nan"),
        "camera_action": "NONE",
        "loiter_time_s": float("nan"),
        "camera_photo_interval_s": float("nan"),
        "acceptance_radius_m": 5.0,
        "yaw_deg": float("nan"),
        "camera_photo_distance_m": float("nan"),
        "vehicle_action": "NONE",
    }


class MavsdkBackend:
    """MAVSDK backend connecting to PX4 SITL.

    Requires: pip install -e .[mavlink]
    """

    def __init__(
        self,
        vehicle_id: str = "uav_001",
        connection_url: str = "udpin://0.0.0.0:14540",
        connect_timeout_s: int = 20,
        default_waypoint_altitude_m: float = 50,
        default_speed_mps: float = 10,
    ) -> None:
        self._vehicle_id = vehicle_id
        self._connection_url = connection_url
        self._connect_timeout_s = connect_timeout_s
        self._default_alt = default_waypoint_altitude_m
        self._default_speed = default_speed_mps
        self._drone: Any = None
        self._connected = False
        self._offboard_active: bool = False
        self._loop = AsyncLoopThread()

    # ------------------------------------------------------------------
    # Lazy MAVSDK import
    # ------------------------------------------------------------------

    def _ensure_mavsdk(self):
        try:
            from mavsdk import System

            return System
        except ImportError as exc:
            raise RuntimeError("mavsdk is not installed. Install with: pip install -e .[mavlink]") from exc

    # ------------------------------------------------------------------
    # Safe call wrapper
    # ------------------------------------------------------------------

    def _safe_call(self, label: str, coro: Any, timeout_s: float = 15.0) -> CommandResult:
        """Run *coro* on the persistent loop, wrapping exceptions."""
        if not self._connected and label != "CONNECT":
            return _result(False, f"{label}: not connected")
        try:
            return self._loop.run(coro, timeout_s=timeout_s)
        except Exception as exc:
            logger.error("{} failed: {}", label, exc)
            return _result(False, f"{label} failed: {exc}")

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _first(stream: Any, timeout_s: float = 2.0) -> Any:
        """Take the first item from an async stream with timeout."""

        async def _get() -> Any:
            async for item in stream:
                return item

        return await asyncio.wait_for(_get(), timeout=timeout_s)

    # ------------------------------------------------------------------
    # connect
    # ------------------------------------------------------------------

    async def _connect_async(self) -> CommandResult:
        System = self._ensure_mavsdk()
        self._drone = System()
        logger.info("MAVSDK connecting to {}...", self._connection_url)
        await self._drone.connect(system_address=self._connection_url)

        try:
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    self._connected = True
                    logger.info("MAVSDK connected to {}", self._connection_url)
                    return _result(True, "Connected to PX4 SITL")
        except TimeoutError:
            return _result(False, f"Connection timeout ({self._connect_timeout_s}s)")

        return _result(False, "Connection failed — no connected state received")

    def connect(self) -> CommandResult:
        return self._safe_call("CONNECT", self._connect_async(), timeout_s=float(self._connect_timeout_s))

    # ------------------------------------------------------------------
    # telemetry
    # ------------------------------------------------------------------

    async def _get_telemetry_async(self) -> TelemetrySnapshot:
        if self._drone is None:
            return TelemetrySnapshot(vehicle_id=self._vehicle_id, timestamp_utc=utc_now_iso())

        armed = False
        mode = "UNKNOWN"
        lat = None
        lon = None
        alt_m = None
        heading = None
        battery = None

        for label, stream_factory in [
            ("armed", self._drone.telemetry.armed),
            ("flight_mode", self._drone.telemetry.flight_mode),
            ("position", self._drone.telemetry.position),
            ("heading", self._drone.telemetry.heading),
            ("battery", self._drone.telemetry.battery),
        ]:
            try:
                val = await self._first(stream_factory(), timeout_s=2.0)
            except Exception:
                continue
            if label == "armed":
                armed = val
            elif label == "flight_mode":
                mode = val.name if hasattr(val, "name") else str(val)
            elif label == "position":
                lat = val.latitude_deg
                lon = val.longitude_deg
                alt_m = val.relative_altitude_m
            elif label == "heading":
                heading = val.heading_deg
            elif label == "battery":
                battery = val.remaining_percent * 100  # may be double-scaled by some PX4 builds

        position = None
        if lat is not None and lon is not None:
            position = GeoPoint(lat=lat, lon=lon, alt_m=alt_m)

        return TelemetrySnapshot(
            vehicle_id=self._vehicle_id,
            timestamp_utc=utc_now_iso(),
            connected=self._connected,
            armed=armed,
            mode=mode,
            position=position,
            heading_deg=heading,
            battery_pct=battery,
        )

    def get_telemetry(self) -> TelemetrySnapshot:
        try:
            return self._loop.run(self._get_telemetry_async(), timeout_s=15.0)
        except Exception:
            return TelemetrySnapshot(
                vehicle_id=self._vehicle_id,
                timestamp_utc=utc_now_iso(),
                connected=self._connected,
            )

    # ------------------------------------------------------------------
    # action commands
    # ------------------------------------------------------------------

    async def _arm_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        await self._drone.action.arm()
        return _result(True, "Armed")

    def arm(self) -> CommandResult:
        return self._safe_call("ARM", self._arm_async())

    async def _disarm_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        await self._drone.action.disarm()
        return _result(True, "Disarmed")

    def disarm(self) -> CommandResult:
        return self._safe_call("DISARM", self._disarm_async())

    async def _takeoff_async(self, altitude_m: float) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        await self._drone.action.set_takeoff_altitude(altitude_m)
        await self._drone.action.takeoff()
        return _result(True, "Takeoff initiated", altitude_m=altitude_m)

    def takeoff(self, altitude_m: float) -> CommandResult:
        return self._safe_call("TAKEOFF", self._takeoff_async(altitude_m))

    async def _land_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        await self._drone.action.land()
        return _result(True, "Land initiated")

    def land(self) -> CommandResult:
        return self._safe_call("LAND", self._land_async())

    async def _return_home_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        await self._drone.action.return_to_launch()
        return _result(True, "Return to launch initiated")

    def return_home(self) -> CommandResult:
        return self._safe_call("RETURN_HOME", self._return_home_async())

    async def _hold_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        try:
            await self._drone.action.hold()
        except AttributeError:
            return _result(False, "HOLD not supported by this MAVSDK version")
        return _result(True, "Hold initiated")

    def hold(self) -> CommandResult:
        return self._safe_call("HOLD", self._hold_async())

    async def _goto_async(self, point: GeoPoint) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        alt = point.alt_m if point.alt_m is not None else self._default_alt
        await self._drone.action.goto_location(
            latitude_deg=point.lat,
            longitude_deg=point.lon,
            altitude_m=alt,
            yaw_deg=0.0,
        )
        return _result(True, "Goto initiated", lat=point.lat, lon=point.lon)

    def goto(self, point: GeoPoint) -> CommandResult:
        return self._safe_call("GOTO", self._goto_async(point))

    # ------------------------------------------------------------------
    # mission
    # ------------------------------------------------------------------

    async def _upload_mission_async(self, plan: MissionPlan) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")

        from mavsdk.mission import MissionItem
        from mavsdk.mission import MissionPlan as SdkMissionPlan

        camera_action_map = {
            "NONE": MissionItem.CameraAction.NONE,
        }
        vehicle_action_map = {
            "NONE": MissionItem.VehicleAction.NONE,
        }

        items = []
        for wp in plan.waypoints:
            vals = _mission_item_values_from_waypoint(
                wp_lat=wp.lat,
                wp_lon=wp.lon,
                wp_alt_m=wp.alt_m,
                default_alt=self._default_alt,
                speed=self._default_speed,
            )
            items.append(
                MissionItem(
                    vals["latitude_deg"],
                    vals["longitude_deg"],
                    vals["relative_altitude_m"],
                    vals["speed_m_s"],
                    vals["is_fly_through"],
                    vals["gimbal_pitch_deg"],
                    vals["gimbal_yaw_deg"],
                    camera_action_map.get(vals["camera_action"], MissionItem.CameraAction.NONE),
                    vals["loiter_time_s"],
                    vals["camera_photo_interval_s"],
                    vals["acceptance_radius_m"],
                    vals["yaw_deg"],
                    vals["camera_photo_distance_m"],
                    vehicle_action_map.get(vals["vehicle_action"], MissionItem.VehicleAction.NONE),
                )
            )

        sdk_plan = SdkMissionPlan(mission_items=items)
        await self._drone.mission.set_return_to_launch_after_mission(True)
        await self._drone.mission.upload_mission(sdk_plan)
        return _result(True, "Mission uploaded", waypoints=len(items))

    def upload_mission(self, plan: MissionPlan) -> CommandResult:
        return self._safe_call("UPLOAD_MISSION", self._upload_mission_async(plan), timeout_s=30.0)

    async def _start_mission_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        await self._drone.mission.start_mission()
        return _result(True, "Mission started")

    def start_mission(self) -> CommandResult:
        return self._safe_call("START_MISSION", self._start_mission_async())

    # ------------------------------------------------------------------
    # offboard control
    # ------------------------------------------------------------------

    async def _offboard_start_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        from mavsdk.offboard import VelocityBodyYawspeed

        await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await self._drone.offboard.start()
        self._offboard_active = True
        return _result(True, "Offboard started")

    def offboard_start(self) -> CommandResult:
        return self._safe_call("OFFBOARD_START", self._offboard_start_async())

    async def _offboard_stop_async(self) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        from mavsdk.offboard import VelocityBodyYawspeed

        await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await self._drone.offboard.stop()
        self._offboard_active = False
        return _result(True, "Offboard stopped")

    def offboard_stop(self) -> CommandResult:
        return self._safe_call("OFFBOARD_STOP", self._offboard_stop_async())

    async def _offboard_set_velocity_body_async(
        self, vx: float, vy: float, vz: float, yawspeed: float,
    ) -> CommandResult:
        if self._drone is None:
            return _result(False, "Not connected")
        from mavsdk.offboard import VelocityBodyYawspeed

        await self._drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, yawspeed),
        )
        return _result(True, "Velocity set", vx=vx, vy=vy, vz=vz, yawspeed=yawspeed)

    def offboard_set_velocity_body(
        self, vx: float, vy: float, vz: float, yawspeed: float,
    ) -> CommandResult:
        return self._safe_call(
            "OFFBOARD_SET_VELOCITY",
            self._offboard_set_velocity_body_async(vx, vy, vz, yawspeed),
            timeout_s=2.0,
        )

    def offboard_is_active(self) -> bool:
        return getattr(self, "_offboard_active", False)

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._connected = False
        self._drone = None
        if not self._loop.closed:
            self._loop.close()
