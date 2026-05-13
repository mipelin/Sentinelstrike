"""Operator Decision Gate demo — perception + tracking + geolocalization + operator gate + TAK."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from sentinel.common.event_bus import EventBus
from sentinel.common.logging import setup_logging
from sentinel.config.loader import load_config
from sentinel.operator.gate import OperatorDecisionGate
from sentinel.operator.policy import OperatorPolicy
from sentinel.operator.simulator import OperatorSimulator
from sentinel.realtime.loop import RealTimeMissionLoop
from sentinel.recorder.recorder import MissionRecorder

app = typer.Typer(help="Operator Decision Gate demo.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    mission_id: str = typer.Option("operator_demo", help="Mission identifier"),
    max_frames: int = typer.Option(30, help="Max frames to process"),
    backend: str = typer.Option("mock", help="Perception backend: mock / yolo"),
    tak_mode: str = typer.Option("dry_run", help="TAK mode: dry_run / udp"),
    operator_mode: str = typer.Option("confirm_above_threshold", help="Simulator mode"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — Operator Decision Gate Demo[/bold green]"))
    console.print(f"  Operator mode: [cyan]{operator_mode}[/cyan]")
    console.print(f"  Backend: [cyan]{backend}[/cyan]")
    console.print(f"  Max frames: [cyan]{max_frames}[/cyan]")

    event_bus = EventBus()

    recorder = MissionRecorder(cfg.recorder.output_dir, mission_id)
    run_dir = recorder.start_run()
    event_bus.subscribe("*", recorder.record_event)

    # Perception backend
    from sentinel.perception.mock_backend import MockPerceptionBackend
    perc_cfg = cfg.perception.model_copy(update={"backend": backend, "max_frames": max_frames})
    if backend == "mock":
        perception_backend_obj = MockPerceptionBackend(
            classes=perc_cfg.classes,
            confidence_threshold=perc_cfg.confidence_threshold,
        )
    else:
        from sentinel.perception.yolo_backend import YoloPerceptionBackend
        perception_backend_obj = YoloPerceptionBackend(
            model_path=perc_cfg.model_path,
            classes=perc_cfg.classes,
            confidence_threshold=perc_cfg.confidence_threshold,
            device=perc_cfg.device,
            image_size=perc_cfg.image_size,
        )

    from sentinel.tracker.runner import TrackingRunner
    tracker_runner = TrackingRunner(config=cfg.tracker, mission_id=mission_id, event_bus=event_bus, run_dir=run_dir)

    from sentinel.geolocalizer.runner import GeolocalizationRunner
    geo_runner = GeolocalizationRunner(config=cfg.geolocalizer, mission_id=mission_id, event_bus=event_bus, run_dir=run_dir)

    from sentinel.tak_bridge.bridge import TakBridge
    tak_bridge_obj = TakBridge(
        config=cfg.tak.model_copy(update={"mode": tak_mode}),
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
    )

    # Operator gate
    op_cfg = cfg.operator
    policy = OperatorPolicy(
        min_track_confidence_for_confirmation=op_cfg.min_track_confidence_for_confirmation,
        min_observation_confidence_for_orbit=op_cfg.min_observation_confidence_for_orbit,
        require_human_for_orbit=op_cfg.require_human_for_orbit,
        require_human_for_abort=op_cfg.require_human_for_abort,
        auto_reject_below_confidence=op_cfg.auto_reject_below_confidence,
    )
    gate = OperatorDecisionGate(
        mission_id=mission_id,
        policy=policy,
        event_bus=event_bus,
        run_dir=run_dir,
        operator_id=op_cfg.operator_id,
    )
    simulator = OperatorSimulator(
        mode=operator_mode,
        operator_id=op_cfg.operator_id,
        policy=policy,
    )

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
        perception_backend=perception_backend_obj,
        tracker_runner=tracker_runner,
        geolocalization_runner=geo_runner,
        tak_bridge=tak_bridge_obj,
        max_frames=max_frames,
        target_fps=10.0,
        video_source=perc_cfg.source,
        operator_gate=gate,
        operator_simulator=simulator,
    )

    try:
        metrics = loop.run()
    finally:
        recorder.close()

    op_metrics_path = run_dir / "operator_metrics.json"
    if op_metrics_path.exists():
        op_metrics = json.loads(op_metrics_path.read_text())
        console.print("\n[bold cyan]Operator Gate:[/bold cyan]")
        console.print(f"  Decisions required: {op_metrics.get('decisions_required', 0)}")
        console.print(f"  Confirmed: {op_metrics.get('confirmed_count', 0)}")
        console.print(f"  Rejected: {op_metrics.get('rejected_count', 0)}")
        console.print(f"  Avg confidence: {op_metrics.get('average_decision_confidence', 0):.3f}")

    console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")
    console.print(f"  Frames: {metrics['frame_count']}")
    console.print(f"  Detections: {metrics['detection_count']}")
    console.print(f"  Tracks: {metrics['track_count']}")
    console.print(f"  Geo observations: {metrics['geo_observation_count']}")
    console.print(f"  TAK messages: {metrics['tak_message_count']}")

    console.print(Panel("[bold green]Operator Decision Gate Demo complete.[/bold green]"))


if __name__ == "__main__":
    app()
