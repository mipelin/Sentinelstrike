"""Edge Agent Runtime — master operational orchestrator."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.geo import haversine_distance_m
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionRequest, VehicleState
from sentinel.config.schema import AppConfig
from sentinel.geolocalizer.runner import GeolocalizationRunner
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
from sentinel.mavlink_bridge.safety import MavlinkSafetyConfig, SafetyPolicy
from sentinel.mavlink_bridge.telemetry import normalize_battery_pct
from sentinel.mission_planner.planner import MissionPlanner, build_sitl_local_request
from sentinel.mission_planner.validation import validate_waypoints_near_home
from sentinel.pipeline.report import write_markdown_report
from sentinel.pipeline.summary import build_pipeline_summary, write_summary_json
from sentinel.recorder.recorder import MissionRecorder
from sentinel.tak_bridge.bridge import TakBridge
from sentinel.tracker.runner import TrackingRunner

from .context import EdgeAgentContext
from .health import write_health_snapshot
from .lifecycle import (
    COMPLETED,
    CONFIG_LOADED,
    FAILED,
    GEOLOCALIZATION_DONE,
    MISSION_PLANNED,
    MISSION_STARTED,
    MISSION_UPLOADED,
    PERCEPTION_RUNNING,
    RECORDER_STARTED,
    REPORT_GENERATED,
    TAK_PUBLISHED,
    TRACKING_RUNNING,
    VEHICLE_CONNECTED,
)
from .modes import EdgeAgentMode, validate_mode

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


def _compute_tracker_metrics(tracks: list) -> dict:
    if not tracks:
        return {"total_tracks": 0, "active_tracks": 0, "lost_tracks": 0, "terminated_tracks": 0}
    return {
        "total_tracks": len(tracks),
        "active_tracks": sum(1 for t in tracks if t.status == "active"),
        "lost_tracks": sum(1 for t in tracks if t.status == "lost"),
        "terminated_tracks": sum(1 for t in tracks if t.status == "terminated"),
    }


class EdgeAgentRuntime:
    def __init__(
        self,
        config: AppConfig,
        mission_request: MissionRequest,
        mode: str = "mock",
        perception_backend: str = "mock",
        tak_mode: str = "dry_run",
        max_frames: int | None = 30,
        local_sitl_mission: bool = True,
        vehicle_lat: float = 38.001,
        vehicle_lon: float = -8.001,
        vehicle_alt: float = 80.0,
        vehicle_heading: float = 90.0,
    ) -> None:
        self._config = config
        self._mission_request = mission_request
        self._mode = validate_mode(mode)
        self._perception_backend = perception_backend
        self._tak_mode = tak_mode
        self._max_frames = max_frames
        self._local_sitl_mission = local_sitl_mission
        self._vehicle_lat = vehicle_lat
        self._vehicle_lon = vehicle_lon
        self._vehicle_alt = vehicle_alt
        self._vehicle_heading = vehicle_heading

    def run(self) -> dict:
        event_bus = EventBus()
        ctx = EdgeAgentContext(
            config=self._config,
            mission_request=self._mission_request,
            event_bus=event_bus,
            mode=self._mode,
        )

        from .lifecycle import EdgeAgentLifecycle

        lc = EdgeAgentLifecycle(mission_id=self._mission_request.mission_id, event_bus=event_bus)

        try:
            return self._run_inner(ctx, lc, event_bus)
        except Exception as exc:
            logger.exception("Edge agent runtime failed")
            lc.transition(FAILED, reason=str(exc))
            self._attempt_safe_land(ctx)
            raise
        finally:
            ctx.close_all()

    def _run_inner(self, ctx: EdgeAgentContext, lc: EdgeAgentLifecycle, event_bus: EventBus) -> dict:  # type: ignore[name-defined]
        mid = self._mission_request.mission_id

        # --- A. Initialization ---
        lc.transition(CONFIG_LOADED, reason="config loaded")
        ctx.started_at_utc = utc_now_iso()

        recorder = MissionRecorder(self._config.recorder.output_dir, mid)
        run_dir = recorder.start_run()
        event_bus.subscribe("*", recorder.record_event)
        ctx.recorder = recorder
        ctx.run_dir = run_dir
        lc.transition(RECORDER_STARTED)

        event_bus.publish(
            make_event(mission_id=mid, source="edge_agent", event_type="edge_agent_started", payload={"mode": self._mode})
        )

        # --- B. Mission planning ---
        mission_request = self._mission_request

        if self._mode == EdgeAgentMode.PX4_SITL and self._local_sitl_mission:
            from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

            sitl_cfg = self._config.sitl_local_mission
            backend = MavsdkBackend(
                vehicle_id=self._config.system.vehicle_id,
                connection_url=self._config.mavlink.connection_url,
                connect_timeout_s=self._config.mavlink.connect_timeout_s,
                default_waypoint_altitude_m=self._config.mavlink.default_waypoint_altitude_m,
                default_speed_mps=self._config.mavlink.default_speed_mps,
            )
            safety_cfg = MavlinkSafetyConfig(
                allow_arm=self._config.mavlink.allow_arm,
                allow_takeoff=self._config.mavlink.allow_takeoff,
                allow_real_backend=self._config.mavlink.allow_real_backend,
                max_takeoff_altitude_m=self._config.mavlink.max_takeoff_altitude_m,
                max_goto_distance_m=self._config.mavlink.max_goto_distance_m,
            )
            safety = SafetyPolicy(safety_cfg, self._config.system.mode)
            temp_bridge = MavlinkBridge(mission_id=mid, vehicle_id=self._config.system.vehicle_id, backend=backend, event_bus=event_bus, safety_policy=safety)

            r = temp_bridge.connect()
            if not r.success:
                raise RuntimeError(f"PX4 SITL connection failed: {r.message}")

            # Poll for home position
            home = None
            for attempt in range(10):
                t = temp_bridge.get_telemetry()
                if t.position and t.position.lat != 0.0 and t.position.lon != 0.0:
                    home = t.position
                    break
                time.sleep(1.0)

            if home is None:
                temp_bridge.close()
                raise RuntimeError("Cannot get valid position from PX4 SITL")

            logger.info("SITL home: lat={:.7f} lon={:.7f}", home.lat, home.lon)
            temp_bridge.close()

            mission_request = build_sitl_local_request(
                home=home,
                size_m=sitl_cfg.size_m,
                altitude_m=sitl_cfg.altitude_m,
                spacing_m=sitl_cfg.spacing_m,
                speed_mps=sitl_cfg.default_speed_mps,
                max_range_km=sitl_cfg.max_range_km,
            )
            logger.info("Mission source: local_sitl_dynamic")

        planner = MissionPlanner()
        mission_plan = planner.create_plan(mission_request)
        ctx.mission_plan = mission_plan
        (run_dir / "mission_plan.json").write_text(mission_plan.model_dump_json(indent=2), encoding="utf-8")
        lc.transition(MISSION_PLANNED, payload={"waypoints": len(mission_plan.waypoints)})

        # --- C. MAVLink ---
        mavlink_results: list[dict] = []

        if self._mode == EdgeAgentMode.MOCK:
            mock_backend = MockMavlinkBackend(vehicle_id=self._config.system.vehicle_id)
            bridge = MavlinkBridge(mission_id=mid, vehicle_id=self._config.system.vehicle_id, backend=mock_backend, event_bus=event_bus)
        elif self._mode == EdgeAgentMode.PX4_SITL:
            if self._config.system.mode != "SIMULATION_MODE":
                raise RuntimeError("PX4 SITL mode requires system.mode=SIMULATION_MODE")

            from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

            px4_backend = MavsdkBackend(
                vehicle_id=self._config.system.vehicle_id,
                connection_url=self._config.mavlink.connection_url,
                connect_timeout_s=self._config.mavlink.connect_timeout_s,
                default_waypoint_altitude_m=self._config.mavlink.default_waypoint_altitude_m,
                default_speed_mps=self._config.mavlink.default_speed_mps,
            )
            safety_cfg = MavlinkSafetyConfig(
                allow_arm=self._config.mavlink.allow_arm,
                allow_takeoff=self._config.mavlink.allow_takeoff,
                allow_real_backend=self._config.mavlink.allow_real_backend,
                max_takeoff_altitude_m=self._config.mavlink.max_takeoff_altitude_m,
                max_goto_distance_m=self._config.mavlink.max_goto_distance_m,
            )
            safety = SafetyPolicy(safety_cfg, self._config.system.mode)
            bridge = MavlinkBridge(mission_id=mid, vehicle_id=self._config.system.vehicle_id, backend=px4_backend, event_bus=event_bus, safety_policy=safety)
        else:
            raise RuntimeError(f"Unsupported mode: {self._mode}")

        ctx.mavlink_bridge = bridge

        r = bridge.connect()
        mavlink_results.append({"command": "connect", "success": r.success, "message": r.message})
        if not r.success:
            raise RuntimeError(f"MAVLink connect failed: {r.message}")
        lc.transition(VEHICLE_CONNECTED)

        # Validate waypoints near home for safety
        telemetry = bridge.get_telemetry()
        if telemetry.position and telemetry.position.lat != 0.0 and telemetry.position.lon != 0.0:
            validate_waypoints_near_home(
                telemetry.position, mission_plan.waypoints, max_distance_m=self._config.sitl_local_mission.max_waypoint_distance_m
            )
            max_dist = max(haversine_distance_m(telemetry.position, wp) for wp in mission_plan.waypoints)
            logger.info("Max waypoint distance from home: {:.0f}m", max_dist)

        r = bridge.upload_mission(mission_plan)
        mavlink_results.append({"command": "upload_mission", "success": r.success, "message": r.message})
        if not r.success:
            raise RuntimeError(f"Mission upload failed: {r.message}")
        lc.transition(MISSION_UPLOADED)

        # Arm/takeoff/start for PX4 SITL; start_mission only for mock
        if self._mode == EdgeAgentMode.PX4_SITL:
            if self._config.mavlink.allow_arm:
                r = bridge.arm()
                mavlink_results.append({"command": "arm", "success": r.success, "message": r.message})
                if not r.success:
                    raise RuntimeError(f"Arm failed: {r.message}")

            if self._config.mavlink.allow_takeoff:
                altitude = min(self._config.mavlink.max_takeoff_altitude_m, 10)
                r = bridge.takeoff(altitude_m=altitude)
                mavlink_results.append({"command": "takeoff", "success": r.success, "message": r.message})
                if not r.success:
                    bridge.land()
                    raise RuntimeError(f"Takeoff failed: {r.message}")

        r = bridge.start_mission()
        mavlink_results.append({"command": "start_mission", "success": r.success, "message": r.message})
        lc.transition(MISSION_STARTED)

        if self._mode == EdgeAgentMode.PX4_SITL:
            telemetry = bridge.get_telemetry()
            logger.info("Post-start telemetry: mode={} armed={}", telemetry.mode, telemetry.armed)

        # --- D. Perception + Tracking ---
        perc_cfg = self._config.perception
        if self._perception_backend:
            perc_cfg = perc_cfg.model_copy(update={"backend": self._perception_backend})
        if self._max_frames is not None:
            perc_cfg = perc_cfg.model_copy(update={"max_frames": self._max_frames})

        perc_source = perc_cfg.source
        if perc_cfg.backend == "mock" and not Path(perc_source).exists():
            perc_source = str(_ensure_synthetic_video(_SYNTHETIC_VIDEO_PATH))
        perc_cfg = perc_cfg.model_copy(update={"source": perc_source})

        tracker_runner = TrackingRunner(config=self._config.tracker, mission_id=mid, event_bus=event_bus, run_dir=run_dir)
        ctx.tracking_runner = tracker_runner
        lc.transition(TRACKING_RUNNING)

        from sentinel.perception.detector import PerceptionRunner

        perc_runner = PerceptionRunner(config=perc_cfg, mission_id=mid, event_bus=event_bus, run_dir=run_dir, tracker_runner=tracker_runner)
        ctx.perception_runner = perc_runner
        lc.transition(PERCEPTION_RUNNING)

        perception_metrics = perc_runner.run()
        all_tracks = tracker_runner.get_all_tracks()

        # --- E. Geolocalization ---
        # Use PX4 telemetry if available, otherwise simulated vehicle state
        if self._mode == EdgeAgentMode.PX4_SITL:
            telemetry = bridge.get_telemetry()
            if telemetry.position and telemetry.position.lat != 0.0:
                vehicle_state = VehicleState(
                    vehicle_id=self._config.system.vehicle_id,
                    timestamp_utc=utc_now_iso(),
                    position=telemetry.position,
                    heading_deg=telemetry.heading_deg or 0.0,
                    groundspeed_mps=telemetry.groundspeed_mps or 0.0,
                    mode=telemetry.mode,
                    armed=telemetry.armed,
                    battery_pct=normalize_battery_pct(telemetry.battery_pct),
                )
            else:
                vehicle_state = self._build_simulated_vehicle_state()
        else:
            vehicle_state = self._build_simulated_vehicle_state()

        ctx.set_vehicle_state(vehicle_state)

        geo_runner = GeolocalizationRunner(config=self._config.geolocalizer, mission_id=mid, event_bus=event_bus, run_dir=run_dir)
        ctx.geolocalization_runner = geo_runner
        observations = geo_runner.process_tracks(all_tracks, vehicle_state)
        geolocalizer_metrics = geo_runner.close()
        lc.transition(GEOLOCALIZATION_DONE)

        # --- F. TAK ---
        tak_cfg = self._config.tak
        if self._tak_mode:
            tak_cfg = tak_cfg.model_copy(update={"mode": self._tak_mode})

        tak_bridge_obj = TakBridge(config=tak_cfg, mission_id=mid, event_bus=event_bus, run_dir=run_dir)
        ctx.tak_bridge = tak_bridge_obj

        tak_bridge_obj.send_vehicle_state(vehicle_state)
        tak_bridge_obj.send_mission_plan(mission_plan)
        for obs in observations:
            tak_bridge_obj.send_geo_observation(obs)

        status_evt = make_event(mission_id=mid, source="edge_agent", event_type="edge_agent_status", payload={"status": "all_phases_complete"})
        tak_bridge_obj.send_system_alert(status_evt, lat=vehicle_state.position.lat if vehicle_state.position else 0, lon=vehicle_state.position.lon if vehicle_state.position else 0)
        tak_metrics = tak_bridge_obj.close()
        lc.transition(TAK_PUBLISHED)

        # --- G. Report / Summary ---
        tracker_metrics = _compute_tracker_metrics(all_tracks)

        # --- H. MAVLink post-mission ---
        if self._mode == EdgeAgentMode.PX4_SITL:
            r = bridge.hold()
            mavlink_results.append({"command": "hold", "success": r.success, "message": r.message})
            r = bridge.return_home()
            mavlink_results.append({"command": "return_home", "success": r.success, "message": r.message})
            r = bridge.land()
            mavlink_results.append({"command": "land", "success": r.success, "message": r.message})

        ctx.ended_at_utc = utc_now_iso()
        lc.transition(COMPLETED)

        summary = build_pipeline_summary(
            mission_id=mid,
            run_dir=run_dir,
            mission_plan=mission_plan,
            perception_metrics=perception_metrics,
            tracker_metrics=tracker_metrics,
            geolocalizer_metrics=geolocalizer_metrics,
            tak_metrics=tak_metrics,
            mavlink_results=mavlink_results,
        )

        summary["edge_agent"] = {
            "mode": self._mode,
            "started_at_utc": ctx.started_at_utc,
            "ended_at_utc": ctx.ended_at_utc,
            "lifecycle_final_state": lc.state,
        }

        write_summary_json(summary, run_dir / "pipeline_summary.json")
        write_markdown_report(summary, run_dir / "report.md")
        write_health_snapshot(ctx, run_dir / "edge_agent_health.json")
        lc.transition(REPORT_GENERATED)

        return summary

    def _build_simulated_vehicle_state(self) -> VehicleState:
        return VehicleState(
            vehicle_id=self._config.system.vehicle_id,
            timestamp_utc=utc_now_iso(),
            position=GeoPoint(lat=self._vehicle_lat, lon=self._vehicle_lon, alt_m=self._vehicle_alt),
            heading_deg=self._vehicle_heading,
            groundspeed_mps=10.0,
            mode="SIMULATED",
            armed=False,
            battery_pct=90.0,
        )

    def _attempt_safe_land(self, ctx: EdgeAgentContext) -> None:
        if ctx.mavlink_bridge is not None and self._mode == EdgeAgentMode.PX4_SITL:
            try:
                ctx.mavlink_bridge.return_home()
                ctx.mavlink_bridge.land()
            except Exception:
                logger.warning("Safe land attempt failed")
