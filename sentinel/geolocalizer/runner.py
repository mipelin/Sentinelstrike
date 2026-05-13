"""Geolocalization runner — wraps geolocalizer with events, throttling, and file output."""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.time import utc_now_iso
from sentinel.common.types import CameraModel, GeoObservation, Track, VehicleState
from sentinel.config.schema import GeolocalizerConfig

from .flat_ground import FlatGroundGeolocalizer
from .metrics import compute_geolocalizer_metrics


class _TrackThrottleState:
    __slots__ = ("last_publish_time", "last_lat", "last_lon", "count")

    def __init__(self) -> None:
        self.last_publish_time: float = 0.0
        self.last_lat: float | None = None
        self.last_lon: float | None = None
        self.count: int = 0


class GeolocalizationRunner:
    def __init__(
        self,
        config: GeolocalizerConfig,
        mission_id: str,
        event_bus: EventBus | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._run_dir = run_dir

        camera = CameraModel(**config.camera) if config.camera else CameraModel()

        self._geolocalizer = FlatGroundGeolocalizer(
            camera=camera,
            assumed_ground_alt_m=config.assumed_ground_alt_m,
            default_accuracy_estimate_m=config.default_accuracy_estimate_m,
        )
        self._observations: list[GeoObservation] = []
        self._obs_file = None
        self._track_throttle: dict[str, _TrackThrottleState] = {}
        self._suppressed_count: int = 0

        if run_dir and config.save_geo_observations_jsonl:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._obs_file = (run_dir / "geo_observations.jsonl").open("w", encoding="utf-8")

    def _publish(self, event_type: str, **payload: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="geolocalizer",
                    event_type=event_type,
                    payload=payload,
                )
            )

    def _should_publish(self, trk: Track, obs: GeoObservation) -> bool:
        if not self._config.publish_active_only:
            return True
        if trk.status != "active":
            return False

        state = self._track_throttle.get(trk.track_id)
        if state is None:
            return True

        if state.count >= self._config.max_observations_per_track:
            return False

        now = time.monotonic()
        elapsed = now - state.last_publish_time
        if elapsed < self._config.min_publish_interval_s:
            return False

        if state.last_lat is not None and state.last_lon is not None:
            dist = _haversine_m(
                state.last_lat, state.last_lon,
                obs.estimated_location.lat, obs.estimated_location.lon,
            )
            if dist < self._config.min_movement_m:
                return False

        return True

    def _record_publish(self, track_id: str, obs: GeoObservation) -> None:
        state = self._track_throttle.get(track_id)
        if state is None:
            state = _TrackThrottleState()
            self._track_throttle[track_id] = state
        state.last_publish_time = time.monotonic()
        state.last_lat = obs.estimated_location.lat
        state.last_lon = obs.estimated_location.lon
        state.count += 1

    def process_tracks(
        self,
        tracks: list[Track],
        vehicle_state: VehicleState,
        timestamp_utc: str | None = None,
        additional_metadata: dict | None = None,
    ) -> list[GeoObservation]:
        ts = timestamp_utc or utc_now_iso()
        results: list[GeoObservation] = []

        for trk in tracks:
            if trk.confidence < self._config.min_track_confidence:
                continue
            if trk.bbox_xyxy is None:
                continue
            if trk.metadata.get("confirmation_state") == "TENTATIVE":
                continue

            obs = self._geolocalizer.estimate_track_location(
                mission_id=self._mission_id,
                track=trk,
                vehicle_state=vehicle_state,
                timestamp_utc=ts,
            )
            if obs is None:
                continue

            # Propagate priority info from track to observation metadata
            track_meta = trk.metadata or {}
            priority_meta = {
                "priority_level": track_meta.get("priority_level", "LOW"),
                "priority_score": track_meta.get("priority_score", 0.0),
                "movement_state": track_meta.get("movement_state", "STATIONARY"),
                "suppressed": track_meta.get("suppressed", False),
                "confirmation_state": track_meta.get("confirmation_state", "TENTATIVE"),
            }
            obs = obs.model_copy(update={"metadata": {**obs.metadata, **priority_meta}})

            if additional_metadata:
                obs = obs.model_copy(update={"metadata": {**obs.metadata, **additional_metadata}})

            if not self._should_publish(trk, obs):
                self._suppressed_count += 1
                continue

            self._record_publish(trk.track_id, obs)
            results.append(obs)
            self._observations.append(obs)

            if self._obs_file:
                self._obs_file.write(obs.model_dump_json() + "\n")

            self._publish(
                "geo_observation_created",
                observation_id=obs.observation_id,
                track_id=obs.track_id,
                class_name=obs.class_name,
                lat=obs.estimated_location.lat,
                lon=obs.estimated_location.lon,
                accuracy_estimate_m=obs.accuracy_estimate_m,
            )

        self._publish("geolocalizer_updated", observation_count=len(results))
        return results

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count

    def close(self) -> dict:
        if self._obs_file:
            self._obs_file.close()

        metrics = compute_geolocalizer_metrics(self._observations)
        metrics["suppressed_geo_updates"] = self._suppressed_count

        if self._run_dir:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            (self._run_dir / "geolocalizer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        self._publish("geolocalization_completed", **metrics)
        logger.info(
            "Geolocalization completed: {} observations, {} suppressed",
            metrics["observation_count"],
            self._suppressed_count,
        )
        return metrics


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
