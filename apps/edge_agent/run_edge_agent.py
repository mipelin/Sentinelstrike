"""Edge Agent Runtime CLI — master operational mission orchestrator.

This is the single entry point for running a complete mission:
planning, MAVLink, perception, tracking, geolocalization, TAK/C2,
reporting and health — in mock or PX4 SITL mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from sentinel.common.logging import setup_logging
from sentinel.common.types import MissionRequest
from sentinel.config.loader import load_config
from sentinel.edge_agent.runtime import EdgeAgentRuntime

app = typer.Typer(help="Edge Agent Runtime — operational mission orchestrator.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    mission: Path = typer.Option("missions/demo_search_area.json", help="Path to mission JSON"),
    mode: str = typer.Option("mock", help="Operating mode: mock / px4_sitl"),
    backend: str = typer.Option("mock", help="Perception backend: mock / yolo"),
    tak_mode: str = typer.Option("dry_run", help="TAK mode: dry_run / udp"),
    max_frames: int = typer.Option(30, help="Max perception frames"),
    local_sitl_mission: bool = typer.Option(
        True,
        "--local-sitl-mission/--no-local-sitl-mission",
        help="Generate local mission around SITL home",
    ),
    vehicle_lat: float = typer.Option(38.001, help="Simulated vehicle latitude"),
    vehicle_lon: float = typer.Option(-8.001, help="Simulated vehicle longitude"),
    vehicle_alt: float = typer.Option(80.0, help="Simulated vehicle altitude (m)"),
    vehicle_heading: float = typer.Option(90.0, help="Simulated vehicle heading (deg)"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — Edge Agent Runtime[/bold green]"))
    console.print(f"  Mode: [cyan]{mode}[/cyan]")
    console.print(f"  Backend: [cyan]{backend}[/cyan]")
    console.print(f"  TAK mode: [cyan]{tak_mode}[/cyan]")

    mission_data = json.loads(mission.read_text(encoding="utf-8"))
    req = MissionRequest(**mission_data)

    runtime = EdgeAgentRuntime(
        config=cfg,
        mission_request=req,
        mode=mode,
        perception_backend=backend,
        tak_mode=tak_mode,
        max_frames=max_frames,
        local_sitl_mission=local_sitl_mission,
        vehicle_lat=vehicle_lat,
        vehicle_lon=vehicle_lon,
        vehicle_alt=vehicle_alt,
        vehicle_heading=vehicle_heading,
    )

    summary = runtime.run()

    run_dir = Path(summary["run_dir"])
    console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")
    console.print(f"  Mode: {summary.get('edge_agent', {}).get('mode', mode)}")
    console.print(f"  Lifecycle: {summary.get('edge_agent', {}).get('lifecycle_final_state', 'unknown')}")
    console.print(f"  Waypoints: {summary['planner']['waypoint_count']}")
    console.print(f"  Detections: {summary['perception']['detection_count']}")
    console.print(f"  Tracks: {summary['tracking']['total_tracks']}")
    console.print(f"  Geo observations: {summary['geolocalization']['observation_count']}")
    console.print(f"  TAK messages: {summary['tak']['message_count']}")
    console.print(f"  Report: {run_dir / 'report.md'}")
    console.print(f"  Health: {run_dir / 'edge_agent_health.json'}")

    console.print(Panel("[bold green]Edge Agent Runtime complete.[/bold green]"))


if __name__ == "__main__":
    app()
