"""Publishing stage — tracks to geo observations and TAK messages."""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.common.types import GeoObservation, Track, VehicleState


@dataclass
class PublishingResult:
    observations: list[GeoObservation] = field(default_factory=list)
    tak_message_count: int = 0


class PublishingStage:
    """Geolocalizes tracks and publishes via TAK."""

    def __init__(
        self,
        geo_runner: object | None = None,
        tak_bridge: object | None = None,
        operator_gate: object | None = None,
        operator_simulator: object | None = None,
    ) -> None:
        self._geo_runner = geo_runner
        self._tak_bridge = tak_bridge
        self._operator_gate = operator_gate
        self._operator_simulator = operator_simulator

    def process(
        self,
        tracks: list[Track],
        vehicle_state: VehicleState | None,
        frame_id: int,
        timestamp_utc: str,
        mission_id: str,
        publish_every_n: int = 1,
    ) -> PublishingResult:
        result = PublishingResult()

        # Geolocalization
        observations: list[GeoObservation] = []
        if self._geo_runner is not None and tracks and vehicle_state is not None:
            try:
                observations = self._geo_runner.process_tracks(
                    tracks=tracks,
                    vehicle_state=vehicle_state,
                    timestamp_utc=timestamp_utc,
                    additional_metadata={"frame_id": frame_id},
                )
            except TypeError:
                observations = self._geo_runner.process_tracks(
                    tracks=tracks,
                    vehicle_state=vehicle_state,
                    timestamp_utc=timestamp_utc,
                )
        result.observations = observations

        # TAK publishing
        if self._tak_bridge is not None:
            for obs in observations:
                confirmed = True
                if self._operator_gate is not None:
                    gate_decisions = self._operator_gate.evaluate_observations([obs], tracks)
                    sim_decision = None
                    if self._operator_simulator is not None:
                        trk_match = next((t for t in tracks if t.track_id == obs.track_id), None)
                        sim_decision = self._operator_simulator.generate_decision(
                            observation=obs, track=trk_match, mission_id=mission_id,
                        )
                    if sim_decision is not None:
                        self._operator_gate.record_decision(sim_decision)
                        confirmed = sim_decision.action.value == "CONFIRM_OBSERVATION"

                if confirmed:
                    ok = self._tak_bridge.send_geo_observation(obs)
                    if ok:
                        result.tak_message_count += 1

            if vehicle_state is not None and (frame_id % max(1, publish_every_n) == 0):
                ok = self._tak_bridge.send_vehicle_state(vehicle_state)
                if ok:
                    result.tak_message_count += 1

        return result
