"""CLI to run perception + tracking + geolocalization pipeline test."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from sentinel.common.event_bus import EventBus
from sentinel.common.logging import setup_logging
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, VehicleState
from sentinel.config.loader import load_config
from sentinel.geolocalizer.runner import GeolocalizationRunner
from sentinel.perception.detector import PerceptionRunner
from sentinel.recorder.recorder import MissionRecorder
from sentinel.tracker.runner import TrackingRunner

app = typer.Typer(help="Run perception + tracking + geolocalization pipeline test.")
console = Console()

_SYNTHETIC_VIDEO_PATH = Path("data/videos/demo.mp4")


def _ensure_synthetic_video(path: Path, num_frames: int = 60) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (w, h))
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx = w // 2 + int(100 * np.sin(i * 0.2))
        cy = h // 2 + int(80 * np.cos(i * 0.15))
        cv2.rectangle(frame, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    logger.info("Created synthetic video: {}", path)
    return path


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    source: str | None = typer.Option(None, help="Override video source path"),
    backend: str | None = typer.Option(None, help="Override backend: mock/yolo"),
    max_frames: int | None = typer.Option(None, help="Override max frames"),
    annotated: bool = typer.Option(False, help="Generate annotated video"),
    vehicle_lat: float = typer.Option(38.001, help="Vehicle latitude"),
    vehicle_lon: float = typer.Option(-8.001, help="Vehicle longitude"),
    vehicle_alt: float = typer.Option(80.0, help="Vehicle altitude (m)"),
    vehicle_heading: float = typer.Option(90.0, help="Vehicle heading (deg)"),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — Geolocalization Test[/bold green]"))

    event_bus = EventBus()
    recorder = MissionRecorder(cfg.recorder.output_dir, cfg.system.mission_id)

    if cfg.recorder.enabled:
        run_dir = recorder.start_run()
        event_bus.subscribe("*", recorder.record_event)
    else:
        run_dir = None

    # Perception + Tracking pipeline
    perc = cfg.perception
    perc_source = source or perc.source
    if backend:
        perc = perc.model_copy(update={"backend": backend})
    if max_frames:
        perc = perc.model_copy(update={"max_frames": max_frames})
    if annotated:
        perc = perc.model_copy(update={"output_annotated_video": True})

    if perc.backend == "mock" and not Path(perc_source).exists():
        perc_source = str(_ensure_synthetic_video(_SYNTHETIC_VIDEO_PATH))
    perc = perc.model_copy(update={"source": perc_source})

    tracker_runner = TrackingRunner(
        config=cfg.tracker,
        mission_id=cfg.system.mission_id,
        event_bus=event_bus,
        run_dir=run_dir if cfg.recorder.enabled else None,
    )

    perc_runner = PerceptionRunner(
        config=perc,
        mission_id=cfg.system.mission_id,
        event_bus=event_bus,
        run_dir=run_dir if cfg.recorder.enabled else None,
        tracker_runner=tracker_runner,
    )

    perc_metrics = perc_runner.run()

    # Geolocalization
    all_tracks = tracker_runner.get_all_tracks()

    vehicle_state = VehicleState(
        vehicle_id=cfg.system.vehicle_id,
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=vehicle_lat, lon=vehicle_lon, alt_m=vehicle_alt),
        heading_deg=vehicle_heading,
        groundspeed_mps=10.0,
        mode="SIMULATED",
        armed=False,
        battery_pct=90.0,
    )

    geo_runner = GeolocalizationRunner(
        config=cfg.geolocalizer,
        mission_id=cfg.system.mission_id,
        event_bus=event_bus,
        run_dir=run_dir if cfg.recorder.enabled else None,
    )

    geo_runner.process_tracks(all_tracks, vehicle_state)
    geo_metrics = geo_runner.close()

    if cfg.recorder.enabled:
        recorder.close()
        console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")
        console.print(f"  frames: {perc_metrics['frame_count']}")
        console.print(f"  detections: {perc_metrics['detection_count']}")
        console.print(f"  tracks: {len(all_tracks)}")
        console.print(f"  geo observations: {geo_metrics['observation_count']}")
        console.print(f"  perception metrics: {run_dir / 'perception_metrics.json'}")  # type: ignore[operator]
        console.print(f"  tracker metrics: {run_dir / 'tracker_metrics.json'}")  # type: ignore[operator]
        console.print(f"  geolocalizer metrics: {run_dir / 'geolocalizer_metrics.json'}")  # type: ignore[operator]

    console.print(Panel("[bold green]Geolocalization test complete.[/bold green]"))


if __name__ == "__main__":
    app()
