"""Tests for the in-process event bus."""

from sentinel.common.event_bus import EventBus
from sentinel.common.events import SystemEvent, make_event


def test_specific_event_type():
    bus = EventBus()
    received: list[SystemEvent] = []
    bus.subscribe("test_event", received.append)

    event = make_event("m1", "test", "test_event")
    bus.publish(event)

    assert len(received) == 1
    assert received[0].event_id == event.event_id


def test_wildcard_receives_all():
    bus = EventBus()
    received: list[SystemEvent] = []
    bus.subscribe("*", received.append)

    bus.publish(make_event("m1", "s", "type_a"))
    bus.publish(make_event("m1", "s", "type_b"))

    assert len(received) == 2


def test_failing_handler_does_not_break_others():
    bus = EventBus()
    ok_received: list[SystemEvent] = []

    def bad_handler(e: SystemEvent) -> None:
        raise RuntimeError("boom")

    bus.subscribe("evt", bad_handler)
    bus.subscribe("evt", ok_received.append)

    bus.publish(make_event("m1", "s", "evt"))

    assert len(ok_received) == 1
