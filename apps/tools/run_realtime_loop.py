"""Real-time mission loop CLI — frame-by-frame observation pipeline.

Runs perception, tracking, geolocalization, and TAK in a real-time loop
reading frames from a video source. Supports mock and PX4 SITL modes.
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from sentinel.common.event_bus import EventBus
from sentinel.common.logging import setup_logging
from sentinel.config.loader import load_config
from sentinel.realtime.loop import RealTimeMissionLoop
from sentinel.recorder.recorder import MissionRecorder

app = typer.Typer(help="Real-Time Mission Loop — frame-by-frame observation pipeline.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/sim.yaml", help="Path to YAML config"),
    mission_id: str = typer.Option("realtime_001", help="Mission identifier"),
    max_frames: int = typer.Option(100, help="Max frames to process"),
    target_fps: float = typer.Option(5.0, help="Target loop rate (Hz)"),
    video_source: str = typer.Option("data/videos/demo.mp4", help="Video source (path or device index)"),
    video_source_type: str = typer.Option("file", help="Video source type: file / webcam / rtsp"),
    webcam_index: int = typer.Option(0, help="Webcam device index (when source type is webcam)"),
    rtsp_url: str = typer.Option("", help="RTSP stream URL (when source type is rtsp)"),
    backend: str = typer.Option("mock", help="Perception backend: mock / yolo"),
    tak_mode: str = typer.Option("dry_run", help="TAK mode: dry_run / udp"),
    mode: str = typer.Option("mock", help="Operating mode: mock / px4_sitl"),
    publish_every_n_frames: int = typer.Option(
        1, help="Publish vehicle state to TAK every N frames",
    ),
) -> None:
    cfg = load_config(config)
    setup_logging(cfg.system.log_level)

    console.print(Panel("[bold green]ONS Sentinel Core — Real-Time Mission Loop[/bold green]"))
    console.print(f"  Mode: [cyan]{mode}[/cyan]")
    console.print(f"  Backend: [cyan]{backend}[/cyan]")
    console.print(f"  TAK mode: [cyan]{tak_mode}[/cyan]")
    console.print(f"  Target FPS: [cyan]{target_fps}[/cyan]")
    console.print(f"  Max frames: [cyan]{max_frames}[/cyan]")
    console.print(f"  Video source: [cyan]{video_source}[/cyan]")
    console.print(f"  Source type: [cyan]{video_source_type}[/cyan]")

    event_bus = EventBus()

    # Set up recorder for event audit trail
    recorder = MissionRecorder(cfg.recorder.output_dir, mission_id)
    run_dir = recorder.start_run()
    event_bus.subscribe("*", recorder.record_event)

    # --- Build subsystems based on mode ---

    perc_cfg = cfg.perception.model_copy(update={
        "backend": backend,
        "max_frames": max_frames,
        "source": video_source,
    })

    # Perception backend (for frame-by-frame processing)
    from sentinel.perception.yolo_backend import YoloPerceptionBackend

    if backend == "mock":
        from sentinel.perception.mock_backend import MockPerceptionBackend
        perception_backend_obj = MockPerceptionBackend(
            classes=perc_cfg.classes,
            confidence_threshold=perc_cfg.confidence_threshold,
        )
    else:
        perception_backend_obj = YoloPerceptionBackend(
            model_path=perc_cfg.model_path,
            classes=perc_cfg.classes,
            confidence_threshold=perc_cfg.confidence_threshold,
            device=perc_cfg.device,
            image_size=perc_cfg.image_size,
        )

    # Tracker
    from sentinel.tracker.runner import TrackingRunner
    tracker_runner = TrackingRunner(
        config=cfg.tracker,
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
    )

    # Geolocalization
    from sentinel.geolocalizer.runner import GeolocalizationRunner
    geo_runner = GeolocalizationRunner(
        config=cfg.geolocalizer,
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
    )

    # TAK
    from sentinel.tak_bridge.bridge import TakBridge
    tak_bridge_obj = TakBridge(
        config=cfg.tak.model_copy(update={"mode": tak_mode}),
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
    )

    # MAVLink (optional, for PX4 SITL telemetry)
    mavlink_bridge = None
    if mode == "px4_sitl":
        from sentinel.mavlink_bridge.bridge import MavlinkBridge
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend
        from sentinel.mavlink_bridge.safety import MavlinkSafetyConfig, SafetyPolicy

        px4_backend = MavsdkBackend(
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
        mavlink_bridge = MavlinkBridge(
            mission_id=mission_id,
            vehicle_id=cfg.system.vehicle_id,
            backend=px4_backend,
            event_bus=event_bus,
            safety_policy=safety,
        )
        r = mavlink_bridge.connect()
        if not r.success:
            recorder.close()
            raise RuntimeError(f"PX4 SITL connection failed: {r.message}")

    # --- Run the loop ---
    video_source_provider = None
    if video_source_type in ("webcam", "rtsp"):
        from sentinel.config.schema import VideoSourceConfig
        from sentinel.sensors.factory import create_video_source

        vs_cfg = VideoSourceConfig(
            source_type=video_source_type,
            webcam_index=webcam_index,
            rtsp_url=rtsp_url,
            target_fps=target_fps,
        )
        video_source_provider = create_video_source(vs_cfg, event_bus=event_bus, mission_id=mission_id)

    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
        mavlink_bridge=mavlink_bridge,
        perception_backend=perception_backend_obj,
        tracker_runner=tracker_runner,
        geolocalization_runner=geo_runner,
        tak_bridge=tak_bridge_obj,
        max_frames=max_frames,
        target_fps=target_fps,
        video_source=video_source,
        publish_every_n_frames=publish_every_n_frames,
        video_source_provider=video_source_provider,
    )

    try:
        metrics = loop.run()
    finally:
        if mavlink_bridge is not None:
            try:
                mavlink_bridge.close()
            except Exception:
                logger.warning("MAVLink bridge close failed")
        recorder.close()

    console.print(f"\n[bold cyan]Run folder:[/bold cyan] {run_dir}")
    console.print(f"  Frames: {metrics['frame_count']}")
    console.print(f"  Detections: {metrics['detection_count']}")
    console.print(f"  Tracks: {metrics['track_count']}")
    console.print(f"  Geo observations: {metrics['geo_observation_count']}")
    console.print(f"  TAK messages: {metrics['tak_message_count']}")
    console.print(f"  Avg loop rate: {metrics['average_loop_hz']} Hz")
    console.print(f"  Avg frame processing: {metrics['average_frame_processing_ms']} ms")
    console.print(f"  Telemetry stale: {metrics['telemetry_stale_count']}")
    if metrics.get("dropped_frames", 0) > 0 or metrics.get("reconnect_count", 0) > 0:
        console.print(f"  Dropped frames: {metrics['dropped_frames']}")
        console.print(f"  Capture FPS: {metrics['capture_fps']}")
        console.print(f"  Reconnects: {metrics['reconnect_count']}")
    console.print(f"  Metrics: {run_dir / 'realtime_metrics.json'}")
    console.print(f"  Events: {run_dir / 'events.jsonl'}")

    console.print(Panel("[bold green]Real-Time Mission Loop complete.[/bold green]"))


if __name__ == "__main__":
    app()
