"""FlightBridgeWorker — isolates all MAVSDK communication behind a worker.

Owns:
  - MAVSDK connection lifecycle
  - Telemetry subscription → TelemetryCache
  - Offboard command loop consuming VelocityCommand from LatestSlot

No other module should import MAVSDK or call offboard/telemetry APIs directly.
The vision/tracker/render loop reads from TelemetryCache snapshots — zero MAVSDK
calls in the hot path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from sentinel.flight.telemetry_cache import TelemetryCache
from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import DroneState, FlightStatus, VelocityCommand
from sentinel.runtime.worker import Worker


class FlightBridgeWorker(Worker):
    """MAVSDK communication worker.

    Subscribes to PX4 telemetry, publishes DroneState into telemetry_slot.
    Reads latest VelocityCommand from command_slot and sends to PX4 at a
    fixed rate. Stale commands expire safely (zero-velocity hold).

    Usage::

        fbw = FlightBridgeWorker(url="udpin://0.0.0.0:14540")
        fbw.connect(timeout_s=20)
        fbw.start()
        ...
        snap = fbw.telemetry_cache.snapshot()
        fbw.command_slot.write(VelocityCommand(vx=0.5, ...))
        fbw.stop()
    """

    def __init__(
        self,
        url: str = "udpin://0.0.0.0:14540",
        *,
        backend: str = "mavsdk",
        mock_connected: bool = False,
        command_stale_s: float = 0.5,
        telemetry_stale_s: float = 5.0,
        name: str = "flight_bridge",
        hz: float = 10.0,
    ) -> None:
        super().__init__(name=name, hz=hz)
        self._url = url
        self._backend_type = backend
        self._command_stale_s = command_stale_s
        self._telemetry_stale_s = telemetry_stale_s

        # Public state
        self.telemetry_cache = TelemetryCache()
        self.telemetry_slot: LatestSlot[DroneState] = LatestSlot()
        self.flight_status_slot: LatestSlot[FlightStatus] = LatestSlot()
        self.command_slot: LatestSlot[VelocityCommand] = LatestSlot()

        # Internal
        self._drone: Any = None
        self._mavsdk_loop: Any = None
        self._connected = False
        self._offboard_engaged = False
        self._autonomy_enabled = False
        self._autonomy_owned_offboard = False
        self._telemetry_tasks: list = []
        self._shutting_down = False

        # Metrics extensions
        self._mavsdk_send_ms: float = 0.0
        self._telemetry_latency_ms: float = 0.0
        self._dropped_stale_commands: int = 0
        self._reconnect_count: int = 0

        # Mock mode
        if backend == "mock":
            self._connected = mock_connected
            self.telemetry_cache.update_connected(mock_connected)

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self, timeout_s: float = 20.0) -> bool:
        """Establish MAVSDK connection. Blocks until connected or timeout."""
        if self._backend_type == "mock":
            logger.info("FlightBridgeWorker: mock mode (no MAVSDK)")
            return self._connected

        try:
            from mavsdk import System
            from sentinel.mavlink_bridge.async_loop import AsyncLoopThread
        except ImportError:
            logger.error("FlightBridgeWorker: mavsdk not available")
            return False

        self._mavsdk_loop = AsyncLoopThread()
        self._drone = System()

        async def _connect() -> bool:
            logger.info("FlightBridgeWorker connecting to {}...", self._url)
            await self._drone.connect(system_address=self._url)
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    return True
            return False

        try:
            connected = self._mavsdk_loop.run(_connect(), timeout_s=timeout_s)
            if connected:
                self._connected = True
                self.telemetry_cache.update_connected(True)
                self._start_telemetry_subscriptions()
                logger.info("FlightBridgeWorker connected to {}", self._url)
                return True
        except Exception as exc:
            logger.error("FlightBridgeWorker connect failed: {}", exc)
        return False

    def _start_telemetry_subscriptions(self) -> None:
        """Start async telemetry subscriptions on the MAVSDK event loop."""
        if self._drone is None or self._mavsdk_loop is None:
            return

        async def _sub_position() -> None:
            try:
                async for pos in self._drone.telemetry.position():
                    self.telemetry_cache.update_position(
                        lat=pos.latitude_deg,
                        lon=pos.longitude_deg,
                        alt_m=pos.relative_altitude_m,
                    )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._shutting_down:
                    logger.debug("FlightBridgeWorker: position subscription error: {}", exc)

        async def _sub_heading() -> None:
            try:
                async for hdg in self._drone.telemetry.heading():
                    self.telemetry_cache.update_heading(hdg.heading_deg)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._shutting_down:
                    logger.debug("FlightBridgeWorker: heading subscription error: {}", exc)

        async def _sub_armed() -> None:
            try:
                async for armed in self._drone.telemetry.armed():
                    self.telemetry_cache.update_armed(armed)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._shutting_down:
                    logger.debug("FlightBridgeWorker: armed subscription error: {}", exc)

        async def _sub_mode() -> None:
            try:
                async for fm in self._drone.telemetry.flight_mode():
                    name = fm.name if hasattr(fm, "name") else str(fm)
                    self.telemetry_cache.update_mode(name)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._shutting_down:
                    logger.debug("FlightBridgeWorker: mode subscription error: {}", exc)

        async def _sub_battery() -> None:
            try:
                async for bat in self._drone.telemetry.battery():
                    self.telemetry_cache.update_battery(bat.remaining_percent * 100)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._shutting_down:
                    logger.debug("FlightBridgeWorker: battery subscription error: {}", exc)

        for coro_factory, name in [
            (_sub_position, "position"),
            (_sub_heading, "heading"),
            (_sub_armed, "armed"),
            (_sub_mode, "mode"),
            (_sub_battery, "battery"),
        ]:
            try:
                task = self._mavsdk_loop.loop.create_task(coro_factory())
                self._telemetry_tasks.append(task)
            except Exception:
                logger.exception("Failed to start telemetry subscription: {}", name)

    # ── Offboard ─────────────────────────────────────────────────────────

    def engage_offboard(self) -> bool:
        """Start offboard mode."""
        if self._backend_type == "mock":
            self._offboard_engaged = True
            return True

        if not self._connected or self._drone is None:
            return False

        try:
            from mavsdk.offboard import VelocityBodyYawspeed

            async def _start() -> bool:
                await self._drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0),
                )
                await self._drone.offboard.start()
                return True

            result = self._mavsdk_loop.run(_start(), timeout_s=5.0)
            if result:
                self._offboard_engaged = True
                logger.info("FlightBridgeWorker: offboard engaged")
            return bool(result)
        except Exception as exc:
            logger.error("FlightBridgeWorker: offboard engage failed: {}", exc)
            return False

    def disengage_offboard(self) -> bool:
        """Stop offboard mode."""
        if self._backend_type == "mock":
            self._offboard_engaged = False
            return True

        self._offboard_engaged = False
        if not self._connected or self._drone is None:
            return False

        try:
            from mavsdk.offboard import VelocityBodyYawspeed

            async def _stop() -> bool:
                await self._drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0),
                )
                await self._drone.offboard.stop()
                return True

            result = self._mavsdk_loop.run(_stop(), timeout_s=5.0)
            if result:
                logger.info("FlightBridgeWorker: offboard disengaged")
            return bool(result)
        except Exception as exc:
            if self._shutting_down:
                logger.debug("FlightBridgeWorker: offboard disengage during shutdown: {}", exc)
            else:
                logger.error("FlightBridgeWorker: offboard disengage failed: {}", exc)
            return False

    @property
    def offboard_active(self) -> bool:
        return self._offboard_engaged

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def autonomy_enabled(self) -> bool:
        return self._autonomy_enabled

    @autonomy_enabled.setter
    def autonomy_enabled(self, value: bool) -> None:
        self._autonomy_enabled = value

    # ── Worker tick ──────────────────────────────────────────────────────

    def tick(self) -> None:
        """Offboard command loop: read latest VelocityCommand, send to PX4."""
        # Publish telemetry snapshot
        self._publish_telemetry()

        # Autonomy-driven offboard lifecycle
        if self._autonomy_enabled and not self._offboard_engaged:
            self.engage_offboard()
            self._autonomy_owned_offboard = True
        elif not self._autonomy_enabled and self._autonomy_owned_offboard:
            self.disengage_offboard()
            self._autonomy_owned_offboard = False

        # Read latest command
        cmd_val, cmd_seq, cmd_ts = self.command_slot.read()

        if not self._offboard_engaged:
            return

        if cmd_val is None:
            return

        # Check command freshness
        cmd_age = time.monotonic() - cmd_ts if cmd_ts > 0 else float("inf")
        if cmd_age > self._command_stale_s:
            self._dropped_stale_commands += 1
            self._metrics.record_skip()
            # Send hold (zero velocity) when command is stale
            self._send_velocity(0.0, 0.0, 0.0, 0.0)
            return

        self._send_velocity(cmd_val.vx, cmd_val.vy, cmd_val.vz, cmd_val.yawspeed)

    def _send_velocity(self, vx: float, vy: float, vz: float, yawspeed: float) -> None:
        """Send velocity command to PX4."""
        if self._backend_type == "mock":
            return

        if not self._connected or self._drone is None:
            return

        try:
            from mavsdk.offboard import VelocityBodyYawspeed

            t0 = time.monotonic()

            async def _send() -> None:
                await self._drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(vx, vy, vz, yawspeed),
                )

            self._mavsdk_loop.run(_send(), timeout_s=2.0)
            self._mavsdk_send_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:
            logger.error("FlightBridgeWorker: velocity send failed: {}", exc)
            self._metrics.record_exception()

    def _publish_telemetry(self) -> None:
        """Publish current telemetry as DroneState contract."""
        from sentinel.common.time import utc_now_iso

        snap = self.telemetry_cache.snapshot()
        state = DroneState(
            seq=self.telemetry_slot.seq_id() + 1,
            ts_monotonic=time.monotonic(),
            ts_utc=utc_now_iso(),
            connected=snap.connected,
            armed=snap.armed,
            mode=snap.mode,
            position=(
                {"lat": snap.position.lat, "lon": snap.position.lon, "alt_m": snap.position.alt_m}
                if snap.position
                else None
            ),
            heading_deg=snap.heading_deg,
            groundspeed_mps=snap.groundspeed_mps,
            battery_pct=snap.battery_pct,
        )
        self.telemetry_slot.write(state)

        flight = FlightStatus(
            ts_monotonic=time.monotonic(),
            ts_utc=utc_now_iso(),
            state=snap.mode if snap.armed else "DISARMED",
            offboard_active=self._offboard_engaged,
        )
        self.flight_status_slot.write(flight)

    # ── Health ───────────────────────────────────────────────────────────

    def health_report(self) -> dict:
        """Extended health report with flight-specific metrics."""
        snap = self.telemetry_cache.snapshot()
        return {
            "connected": self._connected,
            "offboard_active": self._offboard_engaged,
            "telemetry_fresh": not snap.stale(self._telemetry_stale_s),
            "position_age_s": snap.position_age_s,
            "mavsdk_send_ms": round(self._mavsdk_send_ms, 1),
            "dropped_stale_commands": self._dropped_stale_commands,
            "reconnect_count": self._reconnect_count,
            "worker_metrics": self._metrics.snapshot(),
        }

    # ── Cleanup ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop worker and release MAVSDK resources."""
        self._shutting_down = True
        if self._offboard_engaged:
            self.disengage_offboard()
        super().stop()
        # Cancel telemetry subscription tasks
        if self._telemetry_tasks and self._mavsdk_loop is not None:
            async def _cancel_all():
                for task in self._telemetry_tasks:
                    task.cancel()
                await asyncio.gather(*self._telemetry_tasks, return_exceptions=True)
            try:
                self._mavsdk_loop.run(_cancel_all(), timeout_s=3.0)
            except Exception:
                pass
            self._telemetry_tasks.clear()
        if self._mavsdk_loop is not None:
            self._mavsdk_loop.close()
            self._mavsdk_loop = None
        self._drone = None
        self._connected = False
        logger.info("FlightBridgeWorker stopped")
