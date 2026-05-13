"""Jetson Edge Agent CLI — production runtime for Jetson companion computer.

Usage:
    python -m apps.edge_agent.run_jetson_agent \
        --config configs/jetson.yaml \
        --mission missions/demo_search_area.json

Preflight mode:
    python -m apps.edge_agent.run_jetson_agent \
        --config configs/jetson.yaml \
        --dry-run-preflight
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.common.event_bus import EventBus
from sentinel.common.logging import setup_logging
from sentinel.common.types import MissionRequest
from sentinel.config.loader import load_config
from sentinel.deployment.service import JetsonServiceRunner
from sentinel.edge_agent.runtime import EdgeAgentRuntime

app = typer.Typer(help="Jetson Edge Agent — production runtime for NVIDIA Jetson companion computers.")
console = Console()


def _print_profile_table(profile) -> None:
    table = Table(title="System Profile")
    table.add_column("Component", style="cyan")
    table.add_column("Detail", style="green")
    table.add_column("Status")

    table.add_row("Platform", profile.jetson_model or profile.cpu.model, ":heavy_check_mark:" if profile.is_jetson else ":warning:")
    table.add_row("GPU", profile.gpu.name, ":heavy_check_mark:" if profile.gpu.cuda_available else ":x:")
    table.add_row("CUDA", f"CC {profile.gpu.compute_capability}" if profile.gpu.compute_capability else "N/A", ":heavy_check_mark:" if profile.gpu.cuda_available else ":x:")
    table.add_row("CPU Cores", str(profile.cpu.cores), "")
    table.add_row("CPU Freq", f"{profile.cpu.freq_mhz:.0f} MHz" if profile.cpu.freq_mhz else "N/A", "")
    table.add_row("RAM", f"{profile.memory.available_mb} MB free / {profile.memory.total_mb} MB total", ":warning:" if profile.memory.percent_used > 85 else ":heavy_check_mark:")
    table.add_row("Disk", f"{profile.disk.free_mb} MB free / {profile.disk.total_mb} MB total", ":warning:" if profile.disk.percent_used > 90 else ":heavy_check_mark:")
    temp = f"{profile.thermal.cpu_temp_c:.1f} C" if profile.thermal.cpu_temp_c else "N/A"
    table.add_row("CPU Temp", temp, ":warning:" if profile.thermal.cpu_temp_c and profile.thermal.cpu_temp_c > 80 else "")
    if profile.thermal.gpu_temp_c:
        table.add_row("GPU Temp", f"{profile.thermal.gpu_temp_c:.1f} C", ":warning:" if profile.thermal.gpu_temp_c > 80 else "")

    console.print(table)


def _print_preflight_report(report) -> None:
    table = Table(title="Preflight Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    table.add_column("Message")

    _STATUS_DISPLAY = {
        "PASS": ("green", ":heavy_check_mark: PASS"),
        "WARN": ("yellow", ":warning: WARN"),
        "SKIP": ("dim", "[SKIP]"),
        "FAIL": ("red", ":x: FAIL"),
    }

    for c in report.checks:
        style, label = _STATUS_DISPLAY.get(str(c.status), ("", str(c.status)))
        table.add_row(c.name, f"[{style}]{label}[/{style}]", c.message)

    console.print(table)

    if report.recommendations:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for rec in report.recommendations:
            console.print(f"  - {rec}")

    if report.passed:
        console.print("\n[bold green]Preflight passed.[/bold green]")
    else:
        console.print("\n[bold red]Preflight FAILED. Fix errors before running mission.[/bold red]")


@app.command()
def main(
    config: Path = typer.Option("configs/jetson.yaml", help="Path to YAML config"),
    mission: Path = typer.Option("missions/demo_search_area.json", help="Path to mission JSON"),
    backend: str = typer.Option("yolo", help="Perception backend: yolo / mock / tensorrt"),
    tak_mode: str = typer.Option("dry_run", help="TAK mode: dry_run / udp"),
    max_frames: int | None = typer.Option(None, help="Max perception frames (omit for unlimited)"),
    dry_run_preflight: bool = typer.Option(False, "--dry-run-preflight", help="Run preflight checks only, no mission"),
    check_camera: bool = typer.Option(False, "--check-camera", help="Include camera check in preflight"),
    check_mavlink: bool = typer.Option(False, "--check-mavlink", help="Include MAVLink check in preflight"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Strict: warnings become errors"),
    allow_non_jetson: bool = typer.Option(False, "--allow-non-jetson", help="Allow non-Jetson platform"),
    skip_camera: bool = typer.Option(False, "--skip-camera", help="Skip camera check"),
    skip_mavlink: bool = typer.Option(False, "--skip-mavlink", help="Skip MAVLink check"),
    local_sitl_mission: bool = typer.Option(True, "--local-sitl-mission/--no-local-sitl-mission", help="Generate local mission around SITL home (PX4 SITL only)"),
    vehicle_lat: float = typer.Option(38.001, help="Simulated vehicle latitude (mock mode)"),
    vehicle_lon: float = typer.Option(-8.001, help="Simulated vehicle longitude (mock mode)"),
    vehicle_alt: float = typer.Option(80.0, help="Simulated vehicle altitude m (mock mode)"),
    vehicle_heading: float = typer.Option(90.0, help="Simulated vehicle heading deg (mock mode)"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold cyan]ONS Sentinel Core — Jetson Edge Agent[/bold cyan]"))
    console.print(f"  Config: [cyan]{config}[/cyan]")
    console.print(f"  Mode: [cyan]{cfg.system.mode}[/cyan]")
    console.print(f"  Backend: [cyan]{backend}[/cyan]")
    console.print(f"  TAK mode: [cyan]{tak_mode}[/cyan]")
    console.print(f"  Max frames: [cyan]{max_frames if max_frames is not None else 'unlimited'}[/cyan]")

    if hasattr(cfg, "runtime"):
        console.print(f"  Target FPS: [cyan]{cfg.runtime.target_fps}[/cyan]")
        console.print(f"  Save frames: [cyan]{cfg.runtime.save_frames}[/cyan]")
        console.print(f"  Disk budget: [cyan]{cfg.runtime.disk_budget_mb} MB[/cyan]")

    event_bus = EventBus()
    service = JetsonServiceRunner(config=cfg, event_bus=event_bus)

    # Phase 1: Preflight
    mode_is_edge = cfg.system.mode in ("EDGE_MODE", "edge")
    report = service.run_preflight(
        check_camera=check_camera,
        check_mavlink=check_mavlink,
        strict=strict,
        allow_non_jetson=allow_non_jetson,
        skip_camera=skip_camera,
        skip_mavlink=skip_mavlink,
    )
    _print_profile_table(report.profile)
    console.print()
    _print_preflight_report(report)

    if not report.passed:
        raise typer.Exit(code=1)

    if dry_run_preflight:
        console.print("[bold green]Preflight dry-run complete. No mission executed.[/bold green]")
        return

    # Phase 2: Load mission
    if not mission.exists():
        console.print(f"[red]Mission file not found: {mission}[/red]")
        raise typer.Exit(code=1)
    mission_data = json.loads(mission.read_text(encoding="utf-8"))
    req = MissionRequest(**mission_data)

    # Phase 3: Build runtime
    if mode_is_edge:
        console.print("[cyan]Edge mode — connecting to real MAVLink/MAVSDK...[/cyan]")
        mode_str = "px4_sitl"
    else:
        mode_str = cfg.system.mode.lower()

    runtime = EdgeAgentRuntime(
        config=cfg,
        mission_request=req,
        mode=mode_str,
        perception_backend=backend,
        tak_mode=tak_mode,
        max_frames=max_frames,
        local_sitl_mission=local_sitl_mission,
        vehicle_lat=vehicle_lat,
        vehicle_lon=vehicle_lon,
        vehicle_alt=vehicle_alt,
        vehicle_heading=vehicle_heading,
    )

    # Phase 4: Start watchdog
    wd = service.setup_watchdog(mid=req.mission_id)

    # Phase 5: Setup graceful shutdown
    service.setup_signal_handlers()

    # Phase 6: Run mission
    console.print("\n[bold cyan]Starting mission...[/bold cyan]")
    t_start = time.monotonic()

    wd.start()

    try:
        summary = runtime.run()
    except KeyboardInterrupt:
        logger.warning("Mission interrupted by user")
        summary = {"status": "interrupted", "error": "KeyboardInterrupt"}
    except Exception as exc:
        logger.exception("Mission failed")
        summary = {"status": "failed", "error": str(exc)}
    finally:
        wd.stop()

    t_elapsed = time.monotonic() - t_start
    console.print(f"\n[bold]Mission wall time: {t_elapsed:.1f}s[/bold]")

    run_dir = summary.get("run_dir", "")
    if run_dir:
        console.print(f"[bold cyan]Run folder:[/bold cyan] {run_dir}")
    if "error" in summary:
        console.print(f"[bold red]Error: {summary['error']}[/bold red]")
    else:
        console.print(f"  Lifecycle: {summary.get('edge_agent', {}).get('lifecycle_final_state', 'unknown')}")
        console.print(f"  Waypoints: {summary.get('planner', {}).get('waypoint_count', 0)}")
        console.print(f"  Detections: {summary.get('perception', {}).get('detection_count', 0)}")
        console.print(f"  Tracks: {summary.get('tracking', {}).get('total_tracks', 0)}")
        console.print(f"  Geo observations: {summary.get('geolocalization', {}).get('observation_count', 0)}")
        console.print(f"  TAK messages: {summary.get('tak', {}).get('message_count', 0)}")
        if run_dir:
            console.print(f"  Report: {Path(run_dir) / 'report.md'}")
            console.print(f"  Health: {Path(run_dir) / 'edge_agent_health.json'}")

    console.print(Panel("[bold green]Jetson Edge Agent complete.[/bold green]"))


if __name__ == "__main__":
    app()
