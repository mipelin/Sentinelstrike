"""Autonomy supervisor — minimal state machine for mission lifecycle."""

from __future__ import annotations

from enum import StrEnum

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.types import Detection, MissionPlan, OperatorDecision


class State(StrEnum):
    IDLE = "IDLE"
    MISSION_LOADED = "MISSION_LOADED"
    READY = "READY"
    SEARCHING = "SEARCHING"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    TRACKING = "TRACKING"
    AWAITING_OPERATOR_CONFIRMATION = "AWAITING_OPERATOR_CONFIRMATION"
    LOST_LINK = "LOST_LINK"
    RETURN_HOME = "RETURN_HOME"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ERROR = "ERROR"


class AutonomySupervisor:
    def __init__(self, mission_id: str, event_bus: EventBus) -> None:
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._state: State = State.IDLE
        self._plan: MissionPlan | None = None

    @property
    def state(self) -> State:
        return self._state

    def _transition(self, new_state: State, event_type: str, **payload: object) -> None:
        logger.info("[{}] {} -> {} | {}", self._mission_id, self._state, new_state, event_type)
        self._state = new_state
        self._event_bus.publish(
            make_event(
                mission_id=self._mission_id,
                source="autonomy_supervisor",
                event_type=event_type,
                payload=payload,
            )
        )

    def load_mission(self, plan: MissionPlan) -> None:
        self._plan = plan
        self._transition(State.MISSION_LOADED, "mission_loaded")

    def start(self) -> None:
        self._transition(State.SEARCHING, "mission_started")

    def on_detection(self, detection: Detection) -> None:
        self._transition(
            State.OBJECT_DETECTED,
            "object_detected",
            class_name=detection.class_name,
            confidence=detection.confidence,
        )
        self._transition(State.AWAITING_OPERATOR_CONFIRMATION, "awaiting_operator_confirmation")

    def on_operator_decision(self, decision: OperatorDecision) -> None:
        if decision.decision == "confirm_observation":
            self._transition(State.TRACKING, "operator_confirmed", decision_id=decision.decision_id)

    def on_link_lost(self) -> None:
        self._transition(State.LOST_LINK, "link_lost")

    def complete_mission(self) -> None:
        self._transition(State.MISSION_COMPLETE, "mission_complete")
