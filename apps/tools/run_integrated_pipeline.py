"""CLI to run the integrated mission pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from sentinel.common.logging import setup_logging
from sentinel.common.types import MissionRequest
from sentinel.config.loader import load_config
from sentinel.pipeline.integrated_pipeline import IntegratedMissionPipeline

app = typer.Typer(help="Run integrated mission pipeline.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    mission: Path = typer.Option("missions/demo_search_area.json", help="Path to mission JSON"),
    backend: str = typer.Option("mock", help="Perception backend: mock/yolo"),
    max_frames: int = typer.Option(30, help="Max perception frames"),
    tak_mode: str = typer.Option("dry_run", help="TAK mode: dry_run/udp"),
    vehicle_lat: float = typer.Option(38.001, help="Vehicle latitude"),
    vehicle_lon: float = typer.Option(-8.001, help="Vehicle longitude"),
    vehicle_alt: float = typer.Option(80.0, help="Vehicle altitude (m)"),
    vehicle_heading: float = typer.Option(90.0, help="Vehicle heading (deg)"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — Integrated Mission Pipeline[/bold green]"))

    mission_data = json.loads(mission.read_text())
    req = MissionRequest(**mission_data)

    pipeline = IntegratedMissionPipeline(config=cfg, mission_request=req)
    summary = pipeline.run(
        perception_backend_override=backend,
        max_frames_override=max_frames,
        tak_mode_override=tak_mode,
        vehicle_lat=vehicle_lat,
        vehicle_lon=vehicle_lon,
        vehicle_alt_m=vehicle_alt,
        vehicle_heading_deg=vehicle_heading,
    )

    console.print(f"\n[bold cyan]Run folder:[/bold cyan] {summary['run_dir']}")
    console.print(f"  Waypoints: {summary['planner']['waypoint_count']}")
    console.print(f"  Detections: {summary['perception']['detection_count']}")
    console.print(f"  Tracks: {summary['tracking']['total_tracks']}")
    console.print(f"  Geo observations: {summary['geolocalization']['observation_count']}")
    console.print(f"  TAK messages: {summary['tak']['message_count']}")
    console.print(f"  MAVLink commands: {summary['mavlink']['command_count']}")
    console.print(f"  Report: {Path(summary['run_dir']) / 'report.md'}")

    console.print(Panel("[bold green]Integrated pipeline complete.[/bold green]"))


if __name__ == "__main__":
    app()
