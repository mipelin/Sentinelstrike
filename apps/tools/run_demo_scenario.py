"""Demo Scenario Runner CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from sentinel.common.logging import setup_logging
from sentinel.config.loader import load_config
from sentinel.demo.catalog import get_scenario, list_scenarios, scenario_ids
from sentinel.demo.runner import run_scenario

app = typer.Typer(help="Demo Scenario Runner — reproducible mission scenarios.")
console = Console()


@app.command("list")
def list_cmd() -> None:
    """List available demo scenarios."""
    table = Table(title="Demo Scenarios")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("PX4", justify="center")

    for s in list_scenarios():
        px4 = "Yes" if s.requires_px4 else "No"
        table.add_row(s.scenario_id, s.name, s.description[:60] + "...", px4)

    console.print(table)


@app.command("run")
def run_cmd(
    scenario: str = typer.Argument(..., help=f"Scenario ID: {', '.join(scenario_ids())}"),
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
) -> None:
    """Run a specific demo scenario."""
    s = get_scenario(scenario)
    if s is None:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        console.print(f"Available: {', '.join(scenario_ids())}")
        raise typer.Exit(1)

    if s.requires_px4:
        console.print("[yellow]This scenario requires PX4 SITL to be running.[/yellow]")

    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(f"[bold green]Running scenario: {s.name}[/bold green]")
    summary = run_scenario(s, config_path=str(config))

    console.print(f"\n[bold cyan]Run folder:[/bold cyan] {summary['run_dir']}")
    m = summary.get("metrics", {})
    console.print(f"  Frames: {m.get('frame_count', 0)}")
    console.print(f"  Detections: {m.get('detection_count', 0)}")
    console.print(f"  Tracks: {m.get('track_count', 0)}")
    console.print(f"  Observations: {m.get('geo_observation_count', 0)}")
    console.print(f"  Achieved FPS: {m.get('achieved_fps', 0):.1f}")
    console.print(Panel("[bold green]Scenario complete.[/bold green]"))


from rich.panel import Panel


@app.command("run-all-safe")
def run_all_safe(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
) -> None:
    """Run all non-PX4 scenarios."""
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    for s in list_scenarios():
        if s.requires_px4:
            continue
        console.print(f"\n[bold green]Running: {s.name}[/bold green]")
        try:
            summary = run_scenario(s, config_path=str(config))
            console.print(f"  [green]OK[/green] — {summary['run_dir']}")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] — {e}")
            logger.exception("Scenario {} failed", s.scenario_id)

    console.print(Panel("[bold green]All safe scenarios complete.[/bold green]"))


if __name__ == "__main__":
    app()
