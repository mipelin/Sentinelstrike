"""CLI to run a MAVLink mock bridge test."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from sentinel.common.event_bus import EventBus
from sentinel.common.logging import setup_logging
from sentinel.common.types import MissionRequest
from sentinel.config.loader import load_config
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
from sentinel.mavlink_bridge.safety import MavlinkSafetyConfig, SafetyPolicy
from sentinel.mission_planner.planner import MissionPlanner
from sentinel.recorder.recorder import MissionRecorder

app = typer.Typer(help="Run a MAVLink mock bridge test.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    mission: Path = typer.Option("missions/demo_search_area.json", help="Path to mission JSON"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — MAVLink Mock Bridge Test[/bold green]"))

    mission_data = json.loads(mission.read_text(encoding="utf-8"))
    request = MissionRequest(**mission_data)

    event_bus = EventBus()
    recorder = MissionRecorder(cfg.recorder.output_dir, request.mission_id)

    if cfg.recorder.enabled:
        run_dir = recorder.start_run()
        event_bus.subscribe("*", recorder.record_event)

    backend = MockMavlinkBackend(vehicle_id=cfg.system.vehicle_id)
    safety_cfg = MavlinkSafetyConfig(
        allow_arm=cfg.mavlink.allow_arm,
        allow_takeoff=cfg.mavlink.allow_takeoff,
        allow_real_backend=cfg.mavlink.allow_real_backend,
        max_takeoff_altitude_m=cfg.mavlink.max_takeoff_altitude_m,
        max_goto_distance_m=cfg.mavlink.max_goto_distance_m,
    )
    safety = SafetyPolicy(safety_cfg, cfg.system.mode)

    bridge = MavlinkBridge(
        mission_id=request.mission_id,
        vehicle_id=cfg.system.vehicle_id,
        backend=backend,
        event_bus=event_bus,
        safety_policy=safety,
    )

    logger.info("Connecting...")
    bridge.connect()

    logger.info("Reading initial telemetry")
    t = bridge.get_telemetry()
    logger.info("Telemetry: connected={}, armed={}, mode={}", t.connected, t.armed, t.mode)

    planner = MissionPlanner()
    plan = planner.create_plan(request)
    logger.info("Plan: {} waypoints", len(plan.waypoints))

    logger.info("Uploading mission")
    bridge.upload_mission(plan)

    logger.info("Starting mission")
    bridge.start_mission()

    logger.info("Post-mission telemetry")
    t = bridge.get_telemetry()
    logger.info("Telemetry: mode={}, position={}", t.mode, t.position)

    bridge.hold()
    bridge.return_home()
    bridge.land()

    if cfg.recorder.enabled:
        recorder.close()
        console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")

    console.print(Panel("[bold green]MAVLink mock test complete.[/bold green]"))


if __name__ == "__main__":
    app()
