"""Edge agent lifecycle state machine."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event

CREATED = "CREATED"
CONFIG_LOADED = "CONFIG_LOADED"
RECORDER_STARTED = "RECORDER_STARTED"
MISSION_PLANNED = "MISSION_PLANNED"
VEHICLE_CONNECTED = "VEHICLE_CONNECTED"
MISSION_UPLOADED = "MISSION_UPLOADED"
MISSION_STARTED = "MISSION_STARTED"
PERCEPTION_RUNNING = "PERCEPTION_RUNNING"
TRACKING_RUNNING = "TRACKING_RUNNING"
GEOLOCALIZATION_DONE = "GEOLOCALIZATION_DONE"
TAK_PUBLISHED = "TAK_PUBLISHED"
REPORT_GENERATED = "REPORT_GENERATED"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
ABORTED = "ABORTED"

TERMINAL_STATES = {COMPLETED, FAILED, ABORTED}


@dataclass
class Transition:
    from_state: str
    to_state: str
    reason: str | None = None
    payload: dict | None = None


class EdgeAgentLifecycle:
    def __init__(self, mission_id: str, event_bus: EventBus) -> None:
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._state: str = CREATED
        self._history: list[Transition] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def history(self) -> list[Transition]:
        return list(self._history)

    def transition(self, new_state: str, reason: str | None = None, payload: dict | None = None) -> None:
        old = self._state
        self._history.append(Transition(from_state=old, to_state=new_state, reason=reason, payload=payload))
        logger.info("Lifecycle: {} -> {}{}", old, new_state, f" ({reason})" if reason else "")
        self._state = new_state

        self._event_bus.publish(
            make_event(
                mission_id=self._mission_id,
                source="edge_agent_lifecycle",
                event_type="edge_agent_state_changed",
                payload={"from": old, "to": new_state, "reason": reason, "extra": payload or {}},
            )
        )

        if new_state == COMPLETED:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="edge_agent_lifecycle",
                    event_type="edge_agent_completed",
                    payload={"final_state": new_state},
                )
            )
        elif new_state == FAILED:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="edge_agent_lifecycle",
                    event_type="edge_agent_failed",
                    payload={"from_state": old, "reason": reason},
                )
            )
        elif new_state == ABORTED:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="edge_agent_lifecycle",
                    event_type="edge_agent_aborted",
                    payload={"from_state": old, "reason": reason},
                )
            )
