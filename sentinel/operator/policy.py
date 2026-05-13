"""Operator policy — configurable rules for decision gating."""

from __future__ import annotations

from sentinel.common.types import GeoObservation, Track

from .decisions import OperatorAction, PolicyDecision


class OperatorPolicy:
    def __init__(
        self,
        min_track_confidence_for_confirmation: float = 0.6,
        min_observation_confidence_for_orbit: float = 0.6,
        require_human_for_orbit: bool = True,
        require_human_for_abort: bool = False,
        auto_reject_below_confidence: bool = True,
    ) -> None:
        self._min_track_conf = min_track_confidence_for_confirmation
        self._min_obs_conf = min_observation_confidence_for_orbit
        self._require_human_for_orbit = require_human_for_orbit
        self._require_human_for_abort = require_human_for_abort
        self._auto_reject = auto_reject_below_confidence

    def evaluate_observation(
        self,
        track: Track | None = None,
        observation: GeoObservation | None = None,
    ) -> PolicyDecision:
        track_conf = track.confidence if track else 0.0
        obs_conf = observation.confidence if observation else 0.0

        if self._auto_reject and track_conf < self._min_track_conf:
            return PolicyDecision(
                allowed=False,
                required_action=OperatorAction.REJECT_OBSERVATION,
                reason=f"Track confidence {track_conf:.2f} below minimum {self._min_track_conf}",
                severity="warning",
            )

        if self._auto_reject and obs_conf < self._min_obs_conf:
            return PolicyDecision(
                allowed=False,
                required_action=OperatorAction.REJECT_OBSERVATION,
                reason=f"Observation confidence {obs_conf:.2f} below minimum {self._min_obs_conf}",
                severity="warning",
            )

        if track_conf >= self._min_track_conf and obs_conf >= self._min_obs_conf:
            return PolicyDecision(
                allowed=True,
                required_action=OperatorAction.CONFIRM_OBSERVATION,
                reason="Confidence above threshold",
            )

        return PolicyDecision(
            allowed=False,
            required_action=OperatorAction.NO_ACTION,
            reason="No policy match",
        )

    def can_auto_publish(self, track: Track | None = None, observation: GeoObservation | None = None) -> bool:
        track_conf = track.confidence if track else 0.0
        obs_conf = observation.confidence if observation else 0.0
        return track_conf >= self._min_track_conf and obs_conf >= self._min_obs_conf

    def requires_operator_decision(self, track: Track | None = None, observation: GeoObservation | None = None) -> bool:
        track_conf = track.confidence if track else 0.0
        obs_conf = observation.confidence if observation else 0.0

        # Below auto-confirm thresholds requires human
        if track_conf < self._min_track_conf or obs_conf < self._min_obs_conf:
            return True

        # Orbit always requires human if configured
        if self._require_human_for_orbit:
            return True

        return False
