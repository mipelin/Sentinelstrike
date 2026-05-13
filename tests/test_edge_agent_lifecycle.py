"""Tests for EdgeAgentLifecycle."""


from sentinel.common.event_bus import EventBus
from sentinel.edge_agent.lifecycle import (
    COMPLETED,
    CONFIG_LOADED,
    CREATED,
    FAILED,
    EdgeAgentLifecycle,
)


def _collected_events(bus: EventBus) -> list:
    events = []

    def handler(evt):
        events.append(evt)

    bus.subscribe("*", handler)
    return events


def test_initial_state_is_created():
    bus = EventBus()
    lc = EdgeAgentLifecycle(mission_id="t1", event_bus=bus)
    assert lc.state == CREATED


def test_transition_changes_state():
    bus = EventBus()
    lc = EdgeAgentLifecycle(mission_id="t1", event_bus=bus)
    lc.transition(CONFIG_LOADED, reason="config loaded")
    assert lc.state == CONFIG_LOADED


def test_transition_publishes_state_changed_event():
    bus = EventBus()
    events = _collected_events(bus)
    lc = EdgeAgentLifecycle(mission_id="t1", event_bus=bus)
    lc.transition(CONFIG_LOADED)
    types = [e.event_type for e in events]
    assert "edge_agent_state_changed" in types


def test_completed_publishes_completed_event():
    bus = EventBus()
    events = _collected_events(bus)
    lc = EdgeAgentLifecycle(mission_id="t1", event_bus=bus)
    lc.transition(COMPLETED)
    types = [e.event_type for e in events]
    assert "edge_agent_completed" in types


def test_failed_publishes_failed_event():
    bus = EventBus()
    events = _collected_events(bus)
    lc = EdgeAgentLifecycle(mission_id="t1", event_bus=bus)
    lc.transition(FAILED, reason="something broke")
    types = [e.event_type for e in events]
    assert "edge_agent_failed" in types


def test_history_records_transitions():
    bus = EventBus()
    lc = EdgeAgentLifecycle(mission_id="t1", event_bus=bus)
    lc.transition(CONFIG_LOADED, reason="step1")
    lc.transition(COMPLETED, reason="done")
    assert len(lc.history) == 2
    assert lc.history[0].from_state == CREATED
    assert lc.history[0].to_state == CONFIG_LOADED
    assert lc.history[1].from_state == CONFIG_LOADED
    assert lc.history[1].to_state == COMPLETED
