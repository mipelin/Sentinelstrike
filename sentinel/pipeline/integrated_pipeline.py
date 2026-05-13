"""Integrated mission pipeline — runs all subsystems in sequence."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionRequest, VehicleState
from sentinel.config.schema import AppConfig
from sentinel.geolocalizer.runner import GeolocalizationRunner
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
from sentinel.mission_planner.planner import MissionPlanner
from sentinel.perception.detector import PerceptionRunner
from sentinel.recorder.recorder import MissionRecorder
from sentinel.tak_bridge.bridge import TakBridge
from sentinel.tracker.runner import TrackingRunner

from .report import write_markdown_report
from .summary import build_pipeline_summary, write_summary_json

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
    return path


class IntegratedMissionPipeline:
    def __init__(
        self,
        config: AppConfig,
        mission_request: MissionRequest,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._mission_request = mission_request
        self._event_bus = event_bus or EventBus()

    def _publish(self, event_type: str, **payload: object) -> None:
        self._event_bus.publish(
            make_event(
                mission_id=self._mission_request.mission_id,
                source="pipeline",
                event_type=event_type,
                payload=payload,
            )
        )

    def run(
        self,
        perception_backend_override: str | None = None,
        max_frames_override: int | None = None,
        tak_mode_override: str | None = None,
        vehicle_lat: float = 38.001,
        vehicle_lon: float = -8.001,
        vehicle_alt_m: float = 80.0,
        vehicle_heading_deg: float = 90.0,
    ) -> dict:

        try:
            return self._run_inner(
                perception_backend_override=perception_backend_override,
                max_frames_override=max_frames_override,
                tak_mode_override=tak_mode_override,
                vehicle_lat=vehicle_lat,
                vehicle_lon=vehicle_lon,
                vehicle_alt_m=vehicle_alt_m,
                vehicle_heading_deg=vehicle_heading_deg,
            )
        except Exception:
            self._publish("pipeline_failed", error="exception during pipeline execution")
            logger.exception("Pipeline failed")
            raise

    def _run_inner(
        self,
        perception_backend_override: str | None,
        max_frames_override: int | None,
        tak_mode_override: str | None,
        vehicle_lat: float,
        vehicle_lon: float,
        vehicle_alt_m: float,
        vehicle_heading_deg: float,
    ) -> dict:
        mid = self._mission_request.mission_id

        # Recorder
        recorder = MissionRecorder(self._config.recorder.output_dir, mid)
        run_dir = recorder.start_run()
        self._event_bus.subscribe("*", recorder.record_event)

        try:
            self._publish("pipeline_started", mission_id=mid)

            # Mission plan
            planner = MissionPlanner()
            mission_plan = planner.create_plan(self._mission_request)
            (run_dir / "mission_plan.json").write_text(mission_plan.model_dump_json(indent=2), encoding="utf-8")
            self._publish(
                "pipeline_mission_planned",
                waypoints=len(mission_plan.waypoints),
                distance_m=mission_plan.total_distance_m,
            )

            # MAVLink mock
            mavlink_results: list[dict] = []
            mock_backend = MockMavlinkBackend(vehicle_id=self._config.system.vehicle_id)
            bridge = MavlinkBridge(
                mission_id=mid,
                vehicle_id=self._config.system.vehicle_id,
                backend=mock_backend,
                event_bus=self._event_bus,
            )

            for cmd_name, cmd_fn in [
                ("connect", bridge.connect),
                ("upload_mission", lambda: bridge.upload_mission(mission_plan)),
                ("start_mission", bridge.start_mission),
                ("hold", bridge.hold),
                ("return_home", bridge.return_home),
                ("land", bridge.land),
            ]:
                result = cmd_fn()
                mavlink_results.append({"command": cmd_name, "success": result.success, "message": result.message})

            bridge.close()

            # Perception + Tracking
            perc = self._config.perception
            if perception_backend_override:
                perc = perc.model_copy(update={"backend": perception_backend_override})
            if max_frames_override:
                perc = perc.model_copy(update={"max_frames": max_frames_override})

            perc_source = perc.source
            if perc.backend == "mock" and not Path(perc_source).exists():
                perc_source = str(_ensure_synthetic_video(_SYNTHETIC_VIDEO_PATH))
            perc = perc.model_copy(update={"source": perc_source})

            tracker_runner = TrackingRunner(
                config=self._config.tracker,
                mission_id=mid,
                event_bus=self._event_bus,
                run_dir=run_dir,
            )

            perc_runner = PerceptionRunner(
                config=perc,
                mission_id=mid,
                event_bus=self._event_bus,
                run_dir=run_dir,
                tracker_runner=tracker_runner,
            )

            perception_metrics = perc_runner.run()
            all_tracks = tracker_runner.get_all_tracks()

            # Geolocalization
            vehicle_state = VehicleState(
                vehicle_id=self._config.system.vehicle_id,
                timestamp_utc=utc_now_iso(),
                position=GeoPoint(lat=vehicle_lat, lon=vehicle_lon, alt_m=vehicle_alt_m),
                heading_deg=vehicle_heading_deg,
                groundspeed_mps=10.0,
                mode="SIMULATED",
                armed=False,
                battery_pct=90.0,
            )

            geo_runner = GeolocalizationRunner(
                config=self._config.geolocalizer,
                mission_id=mid,
                event_bus=self._event_bus,
                run_dir=run_dir,
            )
            observations = geo_runner.process_tracks(all_tracks, vehicle_state)
            geolocalizer_metrics = geo_runner.close()

            # TAK
            tak_cfg = self._config.tak
            if tak_mode_override:
                tak_cfg = tak_cfg.model_copy(update={"mode": tak_mode_override})

            tak_bridge = TakBridge(
                config=tak_cfg,
                mission_id=mid,
                event_bus=self._event_bus,
                run_dir=run_dir,
            )
            tak_bridge.send_vehicle_state(vehicle_state)
            tak_bridge.send_mission_plan(mission_plan)
            for obs in observations:
                tak_bridge.send_geo_observation(obs)

            status_evt = make_event(
                mission_id=mid,
                source="pipeline",
                event_type="pipeline_status",
                payload={"status": "all_phases_complete"},
            )
            tak_bridge.send_system_alert(status_evt, lat=vehicle_lat, lon=vehicle_lon)
            tak_metrics = tak_bridge.close()

            # Tracker metrics (from close)
            tracker_metrics_data = compute_tracker_metrics_from_tracks(all_tracks)

            # Summary + Report
            summary = build_pipeline_summary(
                mission_id=mid,
                run_dir=run_dir,
                mission_plan=mission_plan,
                perception_metrics=perception_metrics,
                tracker_metrics=tracker_metrics_data,
                geolocalizer_metrics=geolocalizer_metrics,
                tak_metrics=tak_metrics,
                mavlink_results=mavlink_results,
            )
            write_summary_json(summary, run_dir / "pipeline_summary.json")
            write_markdown_report(summary, run_dir / "report.md")

            self._publish("pipeline_completed", **summary)

            return summary
        finally:
            recorder.close()


def compute_tracker_metrics_from_tracks(tracks) -> dict:
    if not tracks:
        return {"total_tracks": 0, "active_tracks": 0, "lost_tracks": 0, "terminated_tracks": 0}
    total = len(tracks)
    active = sum(1 for t in tracks if t.status == "active")
    lost = sum(1 for t in tracks if t.status == "lost")
    terminated = sum(1 for t in tracks if t.status == "terminated")
    return {"total_tracks": total, "active_tracks": active, "lost_tracks": lost, "terminated_tracks": terminated}
