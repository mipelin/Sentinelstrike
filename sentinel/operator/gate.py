"""Operator Decision Gate — evaluates observations against policy and records decisions."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.types import GeoObservation, Track

from .decisions import OperatorAction, OperatorDecisionRecord, make_decision_record
from .metrics import compute_operator_metrics
from .policy import OperatorPolicy


class OperatorDecisionGate:
    def __init__(
        self,
        mission_id: str,
        policy: OperatorPolicy,
        event_bus: EventBus | None = None,
        run_dir: Path | None = None,
        operator_id: str = "local_operator",
    ) -> None:
        self._mission_id = mission_id
        self._policy = policy
        self._event_bus = event_bus
        self._run_dir = run_dir
        self._operator_id = operator_id
        self._decisions: list[OperatorDecisionRecord] = []
        self._pending: list[dict] = []
        self._decisions_file = None
        self._abort_requested = False
        self._return_to_safe_requested = False

        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._decisions_file = (run_dir / "operator_decisions.jsonl").open("w", encoding="utf-8")

    def _publish(self, event_type: str, **payload: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="operator_gate",
                    event_type=event_type,
                    payload=payload,
                )
            )

    def evaluate_observations(
        self,
        observations: list[GeoObservation],
        tracks: list[Track] | None = None,
    ) -> list[OperatorDecisionRecord]:
        track_map: dict[str, Track] = {}
        if tracks:
            for t in tracks:
                track_map[t.track_id] = t

        decisions: list[OperatorDecisionRecord] = []
        for obs in observations:
            track = track_map.get(obs.track_id)
            policy_decision = self._policy.evaluate_observation(track=track, observation=obs)

            if self._policy.requires_operator_decision(track=track, observation=obs):
                self._publish(
                    "operator_decision_required",
                    observation_id=obs.observation_id,
                    track_id=obs.track_id,
                    policy_action=policy_decision.required_action.value,
                    reason=policy_decision.reason,
                )
                self._pending.append({
                    "observation_id": obs.observation_id,
                    "track_id": obs.track_id,
                    "policy_action": policy_decision.required_action.value,
                })

            decisions.append(make_decision_record(
                mission_id=self._mission_id,
                operator_id=self._operator_id,
                action=policy_decision.required_action,
                track_id=obs.track_id,
                observation_id=obs.observation_id,
                reason=policy_decision.reason,
                confidence=obs.confidence,
                source="operator_gate_policy",
            ))

        return decisions

    def record_decision(self, record: OperatorDecisionRecord) -> None:
        self._decisions.append(record)

        if self._decisions_file:
            self._decisions_file.write(record.model_dump_json() + "\n")

        self._publish(
            "operator_decision_recorded",
            decision_id=record.decision_id,
            action=record.action.value,
            observation_id=record.observation_id,
            track_id=record.track_id,
            operator_id=record.operator_id,
        )

        action = record.action
        if action == OperatorAction.CONFIRM_OBSERVATION:
            self._publish(
                "observation_confirmed",
                observation_id=record.observation_id,
                track_id=record.track_id,
                decision_id=record.decision_id,
            )
        elif action == OperatorAction.REJECT_OBSERVATION:
            self._publish(
                "observation_rejected",
                observation_id=record.observation_id,
                track_id=record.track_id,
                decision_id=record.decision_id,
            )
        elif action == OperatorAction.ABORT_MISSION:
            self._abort_requested = True
            self._publish("mission_abort_requested", decision_id=record.decision_id, reason=record.reason)
        elif action == OperatorAction.RETURN_TO_SAFE:
            self._return_to_safe_requested = True
            self._publish("return_to_safe_requested", decision_id=record.decision_id, reason=record.reason)

        # Remove from pending
        self._pending = [
            p for p in self._pending
            if p.get("observation_id") != record.observation_id
        ]

    @property
    def abort_requested(self) -> bool:
        return self._abort_requested

    @property
    def return_to_safe_requested(self) -> bool:
        return self._return_to_safe_requested

    def close(self) -> dict:
        if self._decisions_file:
            self._decisions_file.close()

        metrics = compute_operator_metrics(self._decisions)

        if self._run_dir:
            import json
            self._run_dir.mkdir(parents=True, exist_ok=True)
            (self._run_dir / "operator_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8",
            )

        self._publish("operator_gate_completed", **metrics)
        logger.info(
            "Operator gate: {} decisions ({} confirmed, {} rejected)",
            metrics["decisions_recorded"],
            metrics["confirmed_count"],
            metrics["rejected_count"],
        )
        return metrics

    def get_all_decisions(self) -> list[OperatorDecisionRecord]:
        return list(self._decisions)
