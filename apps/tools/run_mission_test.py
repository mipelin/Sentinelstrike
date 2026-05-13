"""CLI to run a simulated mission end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from sentinel.autonomy_supervisor.supervisor import AutonomySupervisor
from sentinel.common.event_bus import EventBus
from sentinel.common.logging import setup_logging
from sentinel.common.time import utc_now_iso
from sentinel.common.types import BoundingBox, Detection, MissionRequest, OperatorDecision
from sentinel.config.loader import load_config
from sentinel.mission_planner.planner import MissionPlanner
from sentinel.recorder.recorder import MissionRecorder

app = typer.Typer(help="Run a simulated mission test.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    mission: Path = typer.Option("missions/demo_search_area.json", help="Path to mission JSON"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — Mission Test[/bold green]"))

    mission_data = json.loads(mission.read_text(encoding="utf-8"))
    request = MissionRequest(**mission_data)

    event_bus = EventBus()
    recorder = MissionRecorder(cfg.recorder.output_dir, request.mission_id)

    if cfg.recorder.enabled:
        run_dir = recorder.start_run()
        event_bus.subscribe("*", recorder.record_event)

    planner = MissionPlanner()
    plan = planner.create_plan(request)

    logger.info(
        "Plan: {} waypoints, {:.0f}m, {:.1f}s, version={}",
        len(plan.waypoints),
        plan.total_distance_m or 0,
        plan.estimated_duration_s or 0,
        plan.planner_version,
    )

    supervisor = AutonomySupervisor(request.mission_id, event_bus)
    supervisor.load_mission(plan)
    supervisor.start()

    dummy_detection = Detection(
        frame_id=1,
        timestamp_utc=utc_now_iso(),
        class_name="person",
        confidence=0.87,
        bbox_xyxy=BoundingBox(x1=100, y1=150, x2=200, y2=350),
        source="sim_dummy",
    )
    supervisor.on_detection(dummy_detection)

    decision = OperatorDecision(
        decision_id="dec_001",
        mission_id=request.mission_id,
        timestamp_utc=utc_now_iso(),
        operator="local_operator",
        decision="confirm_observation",
    )
    supervisor.on_operator_decision(decision)

    supervisor.complete_mission()

    if cfg.recorder.enabled:
        recorder.close()
        console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")
        console.print(f"  events.jsonl : {run_dir / 'events.jsonl'}")
        console.print(f"  metadata.json: {run_dir / 'metadata.json'}")

    console.print(Panel("[bold green]Mission test complete.[/bold green]"))


if __name__ == "__main__":
    app()
