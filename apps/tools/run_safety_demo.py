"""Safety Action Executor demo — simulates operator abort, link loss, and low battery scenarios."""

from __future__ import annotations

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.common.event_bus import EventBus
from sentinel.common.types import GeoPoint, VehicleState
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
from sentinel.safety.executor import SafetyActionExecutor
from sentinel.safety.metrics import compute_safety_metrics
from sentinel.safety.policy import SafetyPolicyV1

app = typer.Typer(help="Safety Action Executor demo")
console = Console()


def _vehicle(battery: float = 80.0) -> VehicleState:
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc="2026-01-01T00:00:00Z",
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=50.0),
        battery_pct=battery,
    )


def _print_records(records: list, title: str) -> None:
    if not records:
        console.print(f"[dim]No actions for {title}[/dim]")
        return
    table = Table(title=title)
    table.add_column("Record ID", style="dim")
    table.add_column("Trigger", style="bold yellow")
    table.add_column("Action", style="bold cyan")
    table.add_column("Success", style="green")
    table.add_column("Message")
    for r in records:
        success_mark = "[green]OK[/green]" if r.command_success else "[red]FAIL[/red]"
        table.add_row(r.record_id, str(r.trigger), str(r.action), success_mark, r.command_message[:40])
    console.print(table)


@app.command()
def run(
    scenario: str = typer.Argument("all", help="Scenario: abort, link_loss, low_battery, all"),
) -> None:
    """Run safety action executor demo scenarios."""
    console.print(Panel("Safety Action Executor v1 — Demo", style="bold blue"))

    bus = EventBus()
    backend = MockMavlinkBackend(vehicle_id="uav_001")
    bridge = MavlinkBridge(mission_id="demo_001", vehicle_id="uav_001", backend=backend, event_bus=bus)
    policy = SafetyPolicyV1()
    executor = SafetyActionExecutor(mission_id="demo_001", policy=policy, mavlink_bridge=bridge, event_bus=bus)

    if scenario in ("abort", "all"):
        console.print("\n[bold]Scenario: Operator Abort[/bold]")
        recs = executor.evaluate_operator_decision(abort_requested=True, return_to_safe_requested=False)
        _print_records(recs, "Operator Abort")

    if scenario in ("link_loss", "all"):
        executor.reset()
        console.print("\n[bold]Scenario: Link Loss[/bold]")
        recs = executor.evaluate_system_conditions(vehicle_state=_vehicle(), link_lost=True)
        _print_records(recs, "Link Loss")

    if scenario in ("low_battery", "all"):
        executor.reset()
        console.print("\n[bold]Scenario: Low Battery[/bold]")
        recs = executor.evaluate_system_conditions(vehicle_state=_vehicle(battery=12.0))
        _print_records(recs, "Low Battery")

    metrics = compute_safety_metrics(executor.records)
    console.print(Panel(str(metrics), title="Safety Metrics", border_style="green"))
    logger.info("Safety demo complete — {} actions executed", len(executor.records))


if __name__ == "__main__":
    app()
