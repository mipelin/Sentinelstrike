"""Real-time mission loop — frame-by-frame observation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import cv2
from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.types import Detection, GeoObservation, Track, VehicleState
from sentinel.config.schema import AppConfig
from sentinel.geolocalizer.runner import GeolocalizationRunner
from sentinel.perception.drawing import draw_tactical_overlay
from sentinel.tak_bridge.bridge import TakBridge
from sentinel.tracker.runner import TrackingRunner

from .clock import elapsed_s, monotonic_now_s
from .clock import utc_now_iso as _utc_now_iso
from .frame_processor import FrameProcessor
from .metrics import compute_realtime_metrics
from .mock_telemetry import MockMovementTelemetryProvider
from .rate_limiter import RateLimiter
from .telemetry_stream import MavlinkTelemetryProvider

try:
    from sentinel.sensors.managed_source import ManagedVideoSource
except ImportError:
    ManagedVideoSource = None  # type: ignore[assignment,misc]


def _open_video_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _vehicle_state_to_record(vs: VehicleState, stale: bool = False) -> dict:
    pos = vs.position
    return {
        "timestamp_utc": vs.timestamp_utc,
        "vehicle_id": vs.vehicle_id,
        "position": {"lat": pos.lat, "lon": pos.lon, "alt_m": pos.alt_m} if pos else None,
        "heading_deg": vs.heading_deg,
        "groundspeed_mps": vs.groundspeed_mps,
        "battery_pct": vs.battery_pct,
        "mode": vs.mode,
        "armed": vs.armed,
        "stale": stale,
    }


class _TickContext(NamedTuple):
    """Mutable accumulators passed through the loop."""
    frame_count: int
    detection_count: int
    track_count: int
    geo_observation_count: int
    tak_message_count: int
    stale_count: int
    unique_track_ids: set[str]


class RealTimeMissionLoop:
    """Frame-by-frame real-time observation loop.

    Reads frames from a video source, runs perception, tracking,
    geolocalization, and TAK publishing at a configurable tick rate.
    """

    def __init__(
        self,
        config: AppConfig,
        mission_id: str,
        event_bus: EventBus,
        run_dir: Path,
        *,
        mavlink_bridge: object | None = None,
        perception_backend: object | None = None,
        tracker_runner: TrackingRunner | None = None,
        geolocalization_runner: GeolocalizationRunner | None = None,
        tak_bridge: TakBridge | None = None,
        max_frames: int | None = None,
        target_fps: float = 5.0,
        video_source: str = "data/videos/demo.mp4",
        publish_every_n_frames: int = 1,
        operator_gate: object | None = None,
        operator_simulator: object | None = None,
        safety_executor: object | None = None,
        video_source_provider: object | None = None,
        save_frames: bool = True,
        save_latest_frame: bool = True,
        save_annotated_every_n_frames: int = 1,
        disk_budget_mb: int = 2048,
    ) -> None:
        self._config = config
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._run_dir = run_dir
        self._max_frames = max_frames
        self._target_fps = target_fps
        self._video_source = video_source
        self._publish_every_n = max(1, publish_every_n_frames)
        self._video_source_provider = video_source_provider
        self._save_frames = save_frames
        self._save_latest_frame = save_latest_frame
        self._save_annotated_every_n = max(1, save_annotated_every_n_frames)
        self._disk_budget_mb = disk_budget_mb

        if mavlink_bridge is not None:
            self._telemetry_provider = MavlinkTelemetryProvider(
                mavlink_bridge=mavlink_bridge,
                vehicle_id=config.system.vehicle_id,
            )
        else:
            waypoints = self._load_mission_waypoints(config)
            self._telemetry_provider = MockMovementTelemetryProvider(
                vehicle_id=config.system.vehicle_id,
                waypoints=waypoints,
                alt_m=80.0,
                speed_mps=10.0,
            )

        self._frame_processor: FrameProcessor | None = None
        if perception_backend is not None:
            self._frame_processor = FrameProcessor(perception_backend)

        self._tracker_runner = tracker_runner
        self._geo_runner = geolocalization_runner
        self._tak_bridge = tak_bridge

        self._operator_gate = operator_gate
        self._operator_simulator = operator_simulator
        self._safety_executor = safety_executor

        self._rate_limiter = RateLimiter(target_hz=target_fps)

    @staticmethod
    def _load_mission_waypoints(config: AppConfig) -> list:
        """Load mission waypoints for mock movement, if available."""
        try:
            from sentinel.mission_planner.planner import MissionPlanner
        except ImportError:
            return []
        return []

    def run(self) -> dict:
        mid = self._mission_id
        source = "realtime_loop"

        self._event_bus.publish(
            make_event(
                mission_id=mid,
                source=source,
                event_type="realtime_loop_started",
                payload={"target_fps": self._target_fps, "video_source": self._video_source},
            )
        )

        cap: cv2.VideoCapture | None = None
        provider = self._video_source_provider
        use_provider = provider is not None

        if use_provider:
            if not provider.open():
                raise RuntimeError(f"Cannot open video source provider: {provider._source_label}")
        else:
            cap = _open_video_source(self._video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self._video_source}")

        frame_count = 0
        detection_count = 0
        track_count = 0
        geo_observation_count = 0
        tak_message_count = 0
        loop_times_ms: list[float] = []
        stale_count = 0
        unique_track_ids: set[str] = set()

        started_at_utc = _utc_now_iso()
        wall_clock_start = monotonic_now_s()

        det_path = self._run_dir / "detections.jsonl"
        trk_path = self._run_dir / "tracks.jsonl"
        geo_path = self._run_dir / "geo_observations.jsonl"
        tak_path = self._run_dir / "tak_messages.jsonl"
        veh_path = self._run_dir / "vehicle_states.jsonl"
        frames_dir = self._run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        latest_frame_path = self._run_dir / "latest.jpg"
        write_tracks_fallback = not trk_path.exists()
        write_geo_fallback = not geo_path.exists()
        write_tak_fallback = not tak_path.exists()

        try:
            with (
                open(det_path, "a", encoding="utf-8") as det_f,
                open(trk_path, "a", encoding="utf-8") as trk_f,
                open(geo_path, "a", encoding="utf-8") as geo_f,
                open(tak_path, "a", encoding="utf-8") as tak_f,
                open(veh_path, "a", encoding="utf-8") as veh_f,
            ):
                self._rate_limiter.start()
                while True:
                    if use_provider:
                        ret, frame = provider.read_frame()
                    else:
                        ret, frame = cap.read()  # type: ignore[union-attr]
                    if not ret or frame is None:
                        break

                    if self._max_frames is not None and frame_count >= self._max_frames:
                        break

                    tick_start = monotonic_now_s()
                    ts = _utc_now_iso()

                    tick_result = self._tick(
                        frame=frame,
                        frame_id=frame_count,
                        ts=ts,
                        mid=mid,
                        source=source,
                        det_f=det_f,
                        trk_f=trk_f,
                        geo_f=geo_f,
                        tak_f=tak_f,
                        veh_f=veh_f,
                        write_tracks_fallback=write_tracks_fallback,
                        write_geo_fallback=write_geo_fallback,
                        write_tak_fallback=write_tak_fallback,
                        frames_dir=frames_dir,
                        latest_frame_path=latest_frame_path,
                        tick_start=tick_start,
                        wall_clock_start=wall_clock_start,
                        detection_count=detection_count,
                        track_count=track_count,
                        geo_observation_count=geo_observation_count,
                        tak_message_count=tak_message_count,
                        unique_track_ids=unique_track_ids,
                    )

                    detection_count = tick_result["detection_count"]
                    track_count = tick_result["track_count"]
                    geo_observation_count = tick_result["geo_observation_count"]
                    tak_message_count = tick_result["tak_message_count"]
                    loop_times_ms.append(tick_result["tick_ms"])

                    frame_count += 1
                    self._rate_limiter.wait()

        finally:
            if use_provider:
                provider.close()
            elif cap is not None:
                cap.release()

        return self._teardown(
            frame_count=frame_count,
            detection_count=detection_count,
            track_count=track_count,
            geo_observation_count=geo_observation_count,
            tak_message_count=tak_message_count,
            loop_times_ms=loop_times_ms,
            stale_count=stale_count,
            started_at_utc=started_at_utc,
            wall_clock_start=wall_clock_start,
            unique_track_ids=unique_track_ids,
            use_provider=use_provider,
            provider=provider,
        )

    def _tick(
        self,
        *,
        frame,
        frame_id: int,
        ts: str,
        mid: str,
        source: str,
        det_f,
        trk_f,
        geo_f,
        tak_f,
        veh_f,
        write_tracks_fallback: bool,
        write_geo_fallback: bool,
        write_tak_fallback: bool,
        frames_dir: Path,
        latest_frame_path: Path,
        tick_start: float,
        wall_clock_start: float,
        detection_count: int,
        track_count: int,
        geo_observation_count: int,
        tak_message_count: int,
        unique_track_ids: set[str],
    ) -> dict:
        # --- Telemetry ---
        vehicle_state = self._telemetry_provider.get_vehicle_state()
        if isinstance(self._telemetry_provider, MavlinkTelemetryProvider):
            stale_count = self._telemetry_provider.stale_count
        elif isinstance(self._telemetry_provider, MockMovementTelemetryProvider):
            stale_count = self._telemetry_provider.stale_count
        else:
            stale_count = 0

        # --- Vehicle state recording ---
        if vehicle_state is not None:
            stale_flag = stale_count > 0 if isinstance(self._telemetry_provider, MavlinkTelemetryProvider) else False
            veh_record = _vehicle_state_to_record(vehicle_state, stale_flag)
            veh_record["frame_id"] = frame_id
            veh_f.write(json.dumps(veh_record) + "\n")
            if frame_id % 5 == 0:
                self._event_bus.publish(
                    make_event(
                        mission_id=mid,
                        source=source,
                        event_type="vehicle_state_updated",
                        payload=veh_record,
                    )
                )

        # --- Perception ---
        detections: list[Detection] = []
        if self._frame_processor is not None:
            detections = self._frame_processor.process(
                frame, frame_id=frame_id, timestamp_utc=ts,
            )
            detection_count += len(detections)

            for det in detections:
                det_f.write(det.model_dump_json() + "\n")

            if detections:
                self._event_bus.publish(
                    make_event(
                        mission_id=mid,
                        source=source,
                        event_type="realtime_detection",
                        payload={"frame_id": frame_id, "count": len(detections)},
                    )
                )

        # --- Tracking ---
        tracks: list[Track] = []
        if self._tracker_runner is not None:
            tracks = self._tracker_runner.process_frame_detections(
                frame_id=frame_id, timestamp_utc=ts, detections=detections,
            )
            track_count += len(tracks)
            for trk in tracks:
                unique_track_ids.add(trk.track_id)
            if write_tracks_fallback:
                for trk in tracks:
                    trk_f.write(trk.model_dump_json() + "\n")

            if tracks:
                self._event_bus.publish(
                    make_event(
                        mission_id=mid,
                        source=source,
                        event_type="realtime_track_update",
                        payload={"frame_id": frame_id, "count": len(tracks)},
                    )
                )

        # --- Geolocalization ---
        observations: list[GeoObservation] = []
        if self._geo_runner is not None and tracks and vehicle_state is not None:
            try:
                observations = self._geo_runner.process_tracks(
                    tracks=tracks,
                    vehicle_state=vehicle_state,
                    timestamp_utc=ts,
                    additional_metadata={"frame_id": frame_id},
                )
            except TypeError:
                observations = self._geo_runner.process_tracks(
                    tracks=tracks,
                    vehicle_state=vehicle_state,
                    timestamp_utc=ts,
                )
            geo_observation_count += len(observations)
            if write_geo_fallback:
                for obs in observations:
                    geo_f.write(obs.model_dump_json() + "\n")

            if observations:
                self._event_bus.publish(
                    make_event(
                        mission_id=mid,
                        source=source,
                        event_type="realtime_geo_observation",
                        payload={"frame_id": frame_id, "count": len(observations)},
                    )
                )

        # --- TAK ---
        if self._tak_bridge is not None:
            for obs in observations:
                confirmed = True
                if self._operator_gate is not None:
                    self._operator_gate.evaluate_observations([obs], tracks)
                    sim_decision = None
                    if self._operator_simulator is not None:
                        trk_match = next((t for t in tracks if t.track_id == obs.track_id), None)
                        sim_decision = self._operator_simulator.generate_decision(
                            observation=obs, track=trk_match, mission_id=mid,
                        )
                    if sim_decision is not None:
                        self._operator_gate.record_decision(sim_decision)
                        confirmed = sim_decision.action.value == "CONFIRM_OBSERVATION"

                if confirmed:
                    ok = self._tak_bridge.send_geo_observation(obs)
                    if ok:
                        tak_message_count += 1
                        if write_tak_fallback:
                            tak_f.write(json.dumps({
                                "timestamp_utc": ts,
                                "category": "observation",
                                "track_id": obs.track_id,
                                "observation_id": obs.observation_id,
                            }) + "\n")

            if vehicle_state is not None and (frame_id % self._publish_every_n == 0):
                ok = self._tak_bridge.send_vehicle_state(vehicle_state)
                if ok:
                    tak_message_count += 1
                    if write_tak_fallback:
                        tak_f.write(json.dumps({
                            "timestamp_utc": ts,
                            "category": "vehicle",
                            "vehicle_id": vehicle_state.vehicle_id,
                            "frame_id": frame_id,
                        }) + "\n")

            if observations or (frame_id % self._publish_every_n == 0 and vehicle_state):
                self._event_bus.publish(
                    make_event(
                        mission_id=mid,
                        source=source,
                        event_type="realtime_tak_publish",
                        payload={"frame_id": frame_id, "tak_messages": tak_message_count},
                    )
                )

        # --- Safety ---
        if self._safety_executor is not None:
            abort_req = False
            rts_req = False
            if self._operator_gate is not None:
                abort_req = self._operator_gate.abort_requested
                rts_req = self._operator_gate.return_to_safe_requested
            self._safety_executor.evaluate_operator_decision(abort_req, rts_req)
            self._safety_executor.evaluate_system_conditions(
                vehicle_state=vehicle_state,
                stale_count=stale_count,
            )

        # --- Timing ---
        tick_ms = elapsed_s(tick_start) * 1000.0
        monotonic_elapsed_s = elapsed_s(wall_clock_start)

        self._event_bus.publish(
            make_event(
                mission_id=mid,
                source=source,
                event_type="realtime_frame_processed",
                payload={
                    "frame_id": frame_id,
                    "detections": len(detections),
                    "tracks": len(tracks),
                    "observations": len(observations),
                    "tick_ms": round(tick_ms, 2),
                    "monotonic_elapsed_s": round(monotonic_elapsed_s, 4),
                },
            )
        )

        # --- Frame archival ---
        annotated = draw_tactical_overlay(
            frame,
            frame_id=frame_id,
            timestamp_utc=ts,
            detections=detections,
            tracks=tracks,
            vehicle_state=vehicle_state,
        )
        if self._save_frames and (frame_id % self._save_annotated_every_n == 0):
            cv2.imwrite(str(frames_dir / f"frame_{frame_id:06d}.jpg"), annotated)
        if self._save_latest_frame:
            cv2.imwrite(str(latest_frame_path), annotated)

        return {
            "detection_count": detection_count,
            "track_count": track_count,
            "geo_observation_count": geo_observation_count,
            "tak_message_count": tak_message_count,
            "tick_ms": tick_ms,
        }

    def _teardown(
        self,
        *,
        frame_count: int,
        detection_count: int,
        track_count: int,
        geo_observation_count: int,
        tak_message_count: int,
        loop_times_ms: list[float],
        stale_count: int,
        started_at_utc: str,
        wall_clock_start: float,
        unique_track_ids: set[str],
        use_provider: bool,
        provider: object | None,
    ) -> dict:
        mid = self._mission_id
        source = "realtime_loop"

        ended_at_utc = _utc_now_iso()
        wall_clock_s = elapsed_s(wall_clock_start)

        if self._tracker_runner is not None:
            self._tracker_runner.close(frame_count=frame_count)
        if self._geo_runner is not None:
            self._geo_runner.close()
        if self._tak_bridge is not None:
            self._tak_bridge.close()
        if self._operator_gate is not None:
            self._operator_gate.close()
        if self._safety_executor is not None:
            self._safety_executor.close()

        # Sensor health metrics
        _dropped = 0
        _capture_fps = 0.0
        _reconnects = 0
        if use_provider:
            h = provider.health
            _dropped = h.dropped_frames
            _capture_fps = h.fps_estimate
            _reconnects = h.reconnect_count
            provider.write_health_json(self._run_dir / "sensor_health.json")

        _suppressed_geo = 0
        if self._geo_runner is not None:
            _suppressed_geo = getattr(self._geo_runner, "suppressed_count", 0)
        _suppressed_tak = 0
        if self._tak_bridge is not None:
            _suppressed_tak = getattr(self._tak_bridge, "suppressed_count", 0)

        metrics = compute_realtime_metrics(
            frame_count=frame_count,
            detection_count=detection_count,
            track_count=track_count,
            geo_observation_count=geo_observation_count,
            tak_message_count=tak_message_count,
            loop_times_ms=loop_times_ms,
            telemetry_stale_count=stale_count,
            started_at_utc=started_at_utc,
            ended_at_utc=ended_at_utc,
            wall_clock_s=wall_clock_s,
            target_fps=self._target_fps,
            dropped_frames=_dropped,
            capture_fps=_capture_fps,
            reconnect_count=_reconnects,
            unique_tracks=len(unique_track_ids),
            suppressed_geo_updates=_suppressed_geo,
            suppressed_tak_messages=_suppressed_tak,
        )

        metrics_path = self._run_dir / "realtime_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        self._event_bus.publish(
            make_event(
                mission_id=mid,
                source=source,
                event_type="realtime_loop_completed",
                payload=metrics,
            )
        )

        logger.info(
            "Real-time loop done: {} frames, {} detections, {} unique tracks ({} updates), {} geo_obs ({} suppressed), {} TAK msgs ({} suppressed), {:.1f} Hz",
            frame_count, detection_count, len(unique_track_ids), track_count,
            geo_observation_count, _suppressed_geo,
            tak_message_count, _suppressed_tak,
            metrics["average_loop_hz"],
        )

        return metrics
