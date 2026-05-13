"""Operator simulator — generates automated decisions for demos and tests."""

from __future__ import annotations

from sentinel.common.types import GeoObservation, Track

from .decisions import OperatorAction, OperatorDecisionRecord, make_decision_record
from .policy import OperatorPolicy


class OperatorSimulator:
    def __init__(
        self,
        mode: str = "confirm_above_threshold",
        operator_id: str = "sim_operator",
        policy: OperatorPolicy | None = None,
    ) -> None:
        self._mode = mode
        self._operator_id = operator_id
        self._policy = policy or OperatorPolicy()

    @property
    def mode(self) -> str:
        return self._mode

    def generate_decision(
        self,
        observation: GeoObservation,
        track: Track | None = None,
        mission_id: str = "sim",
    ) -> OperatorDecisionRecord | None:
        if self._mode == "auto_confirm":
            return make_decision_record(
                mission_id=mission_id,
                operator_id=self._operator_id,
                action=OperatorAction.CONFIRM_OBSERVATION,
                track_id=observation.track_id,
                observation_id=observation.observation_id,
                reason="auto_confirm",
                confidence=observation.confidence,
                source="operator_simulator",
            )

        if self._mode == "auto_reject":
            return make_decision_record(
                mission_id=mission_id,
                operator_id=self._operator_id,
                action=OperatorAction.REJECT_OBSERVATION,
                track_id=observation.track_id,
                observation_id=observation.observation_id,
                reason="auto_reject",
                confidence=observation.confidence,
                source="operator_simulator",
            )

        if self._mode == "confirm_above_threshold":
            if self._policy.can_auto_publish(track=track, observation=observation):
                return make_decision_record(
                    mission_id=mission_id,
                    operator_id=self._operator_id,
                    action=OperatorAction.CONFIRM_OBSERVATION,
                    track_id=observation.track_id,
                    observation_id=observation.observation_id,
                    reason="confidence_above_threshold",
                    confidence=observation.confidence,
                    source="operator_simulator",
                )
            else:
                return make_decision_record(
                    mission_id=mission_id,
                    operator_id=self._operator_id,
                    action=OperatorAction.REJECT_OBSERVATION,
                    track_id=observation.track_id,
                    observation_id=observation.observation_id,
                    reason="confidence_below_threshold",
                    confidence=observation.confidence,
                    source="operator_simulator",
                )

        if self._mode == "no_action":
            return None

        return None
