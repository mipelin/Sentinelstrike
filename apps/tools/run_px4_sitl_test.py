"""CLI to run a PX4 SITL integration test via MAVSDK.

PX4 SITL must already be running before executing this command.
This command is for simulation only — do NOT use with real hardware.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from sentinel.common.event_bus import EventBus
from sentinel.common.geo import haversine_distance_m
from sentinel.common.logging import setup_logging
from sentinel.common.types import MissionRequest
from sentinel.config.loader import load_config
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend
from sentinel.mavlink_bridge.safety import MavlinkSafetyConfig, SafetyPolicy
from sentinel.mission_planner.planner import MissionPlanner, build_sitl_local_request
from sentinel.mission_planner.validation import validate_waypoints_near_home
from sentinel.recorder.recorder import MissionRecorder

app = typer.Typer(help="Run PX4 SITL integration test. Requires PX4 SITL running.")
console = Console()


def _abort(msg: str, recorder: MissionRecorder | None = None, bridge: MavlinkBridge | None = None) -> None:
    console.print(f"[bold red]{msg}[/bold red]")
    if bridge:
        bridge.close()
    if recorder:
        recorder.close()
    raise typer.Exit(code=1)


def _poll_home_position(bridge: MavlinkBridge, max_attempts: int = 10) -> GeoPoint:  # noqa: F821
    """Poll telemetry until a valid GPS position is received from PX4 SITL."""
    for attempt in range(max_attempts):
        t = bridge.get_telemetry()
        if t.position is not None and t.position.lat != 0.0 and t.position.lon != 0.0:
            return t.position
        logger.debug("Telemetry position not yet valid (attempt {}/{})", attempt + 1, max_attempts)
        time.sleep(1.0)
    _abort("Cannot get valid position from PX4 SITL after {} attempts", bridge=bridge)


@app.command()
def main(
    config: Path = typer.Option("configs/sim_px4.yaml", help="Path to YAML config"),
    mission: Path = typer.Option("missions/demo_search_area.json", help="Path to mission JSON"),
    local_sitl_mission: bool = typer.Option(
        True,
        "--local-sitl-mission/--no-local-sitl-mission",
        help="Generate local mission around SITL home (recommended)",
    ),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold yellow]ONS Sentinel Core — PX4 SITL Integration Test[/bold yellow]"))
    console.print(f"  Connection URL: [cyan]{cfg.mavlink.connection_url}[/cyan]")
    console.print("[bold yellow]PX4 SITL must already be running. This command is for simulation only.[/bold yellow]")

    if cfg.system.mode != "SIMULATION_MODE":
        _abort("ABORT: system.mode must be SIMULATION_MODE for SITL tests.")

    if cfg.mavlink.backend != "mavsdk":
        _abort("ABORT: mavlink.backend must be 'mavsdk' for this test.")

    sitl_cfg = cfg.sitl_local_mission
    if not local_sitl_mission:
        mission_data = json.loads(mission.read_text(encoding="utf-8"))
        request = MissionRequest(**mission_data)

    event_bus = EventBus()
    mission_id = "sitl_local" if local_sitl_mission else getattr(request, "mission_id", "px4_demo_001")
    recorder = MissionRecorder(cfg.recorder.output_dir, mission_id)

    if cfg.recorder.enabled:
        run_dir = recorder.start_run()
        event_bus.subscribe("*", recorder.record_event)
    else:
        run_dir = None

    backend = MavsdkBackend(
        vehicle_id=cfg.system.vehicle_id,
        connection_url=cfg.mavlink.connection_url,
        connect_timeout_s=cfg.mavlink.connect_timeout_s,
        default_waypoint_altitude_m=cfg.mavlink.default_waypoint_altitude_m,
        default_speed_mps=cfg.mavlink.default_speed_mps,
    )

    safety_cfg = MavlinkSafetyConfig(
        allow_arm=cfg.mavlink.allow_arm,
        allow_takeoff=cfg.mavlink.allow_takeoff,
        allow_real_backend=cfg.mavlink.allow_real_backend,
        max_takeoff_altitude_m=cfg.mavlink.max_takeoff_altitude_m,
        max_goto_distance_m=cfg.mavlink.max_goto_distance_m,
    )
    safety = SafetyPolicy(safety_cfg, cfg.system.mode)

    bridge = MavlinkBridge(
        mission_id=mission_id,
        vehicle_id=cfg.system.vehicle_id,
        backend=backend,
        event_bus=event_bus,
        safety_policy=safety,
    )

    # --- connect ---
    logger.info("Connecting to PX4 SITL at {}...", cfg.mavlink.connection_url)
    r = bridge.connect()
    if not r.success:
        _abort(f"Connection failed: {r.message}", recorder=recorder if cfg.recorder.enabled else None, bridge=bridge)

    # --- read home position ---
    logger.info("Reading SITL home position...")
    home = _poll_home_position(bridge)
    logger.info("SITL home position: lat={:.7f} lon={:.7f} alt={:.1f}m", home.lat, home.lon, home.alt_m or 0.0)
    console.print(f"  SITL home: [green]{home.lat:.7f}, {home.lon:.7f}[/green]")

    # --- build mission ---
    if local_sitl_mission:
        request = build_sitl_local_request(
            home=home,
            size_m=sitl_cfg.size_m,
            altitude_m=sitl_cfg.altitude_m,
            spacing_m=sitl_cfg.spacing_m,
            speed_mps=sitl_cfg.default_speed_mps,
            max_range_km=sitl_cfg.max_range_km,
        )
        logger.info("Mission source: local_sitl_dynamic")
        console.print(f"  Mission source: [cyan]local_sitl_dynamic[/cyan] ({sitl_cfg.size_m}m x {sitl_cfg.size_m}m)")
    else:
        logger.info("Mission source: mission_json ({})", mission)
        console.print(f"  Mission source: [cyan]mission_json[/cyan] ({mission})")

    # --- upload mission ---
    planner = MissionPlanner()
    plan = planner.create_plan(request)
    logger.info("Plan: {} waypoints", len(plan.waypoints))

    # --- validate waypoints near home ---
    max_dist = 0.0
    for wp in plan.waypoints:
        d = haversine_distance_m(home, wp)
        if d > max_dist:
            max_dist = d
    logger.info("Max distance from home: {:.0f}m", max_dist)

    try:
        validate_waypoints_near_home(home, plan.waypoints, max_distance_m=sitl_cfg.max_waypoint_distance_m)
    except ValueError as exc:
        _abort(str(exc), recorder=recorder if cfg.recorder.enabled else None, bridge=bridge)

    logger.info("Uploading mission")
    r = bridge.upload_mission(plan)
    logger.info("Upload: success={} msg={}", r.success, r.message)
    if not r.success:
        _abort(f"Upload failed: {r.message}", recorder=recorder if cfg.recorder.enabled else None, bridge=bridge)

    # --- arm ---
    if cfg.mavlink.allow_arm:
        logger.info("Arming")
        r = bridge.arm()
        logger.info("Arm: success={} msg={}", r.success, r.message)
        if not r.success:
            _abort(f"Arm failed: {r.message}", recorder=recorder if cfg.recorder.enabled else None, bridge=bridge)

    # --- takeoff ---
    if cfg.mavlink.allow_takeoff:
        altitude = min(cfg.mavlink.max_takeoff_altitude_m, 10)
        logger.info("Taking off to {}m", altitude)
        r = bridge.takeoff(altitude_m=altitude)
        logger.info("Takeoff: success={} msg={}", r.success, r.message)
        if not r.success:
            logger.warning("Takeoff failed — attempting land before exit")
            bridge.land()
            _abort(f"Takeoff failed: {r.message}", recorder=recorder if cfg.recorder.enabled else None, bridge=bridge)

    # --- mid-flight telemetry ---
    logger.info("Reading mid-flight telemetry")
    t = bridge.get_telemetry()
    logger.info("Telemetry: mode={} armed={}", t.mode, t.armed)

    # --- start mission ---
    logger.info("Starting mission")
    r = bridge.start_mission()
    logger.info("Start mission: success={} msg={}", r.success, r.message)

    # --- post-mission ---
    logger.info("Reading post-mission telemetry")
    t = bridge.get_telemetry()
    logger.info("Telemetry: mode={} position={}", t.mode, t.position)

    r = bridge.hold()
    logger.info("Hold: success={} msg={}", r.success, r.message)

    r = bridge.return_home()
    logger.info("Return home: success={} msg={}", r.success, r.message)

    r = bridge.land()
    logger.info("Land: success={} msg={}", r.success, r.message)

    bridge.close()

    if cfg.recorder.enabled:
        recorder.close()
        console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")

    console.print(Panel("[bold green]PX4 SITL test complete.[/bold green]"))


if __name__ == "__main__":
    app()
