"""Tests for the MAVLink bridge with mock backend."""

from sentinel.common.event_bus import EventBus
from sentinel.common.types import GeoPoint, MissionPlan
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend


def _plan():
    return MissionPlan(
        mission_id="test",
        waypoints=[
            GeoPoint(lat=38.0, lon=-8.0),
            GeoPoint(lat=38.001, lon=-8.0),
            GeoPoint(lat=38.0, lon=-8.0),
        ],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38.0, lon=-8.0),
    )


def _bridge(event_bus: EventBus | None = None) -> MavlinkBridge:
    backend = MockMavlinkBackend()
    return MavlinkBridge(
        mission_id="test",
        vehicle_id="uav_001",
        backend=backend,
        event_bus=event_bus,
    )


def test_connect():
    bridge = _bridge()
    result = bridge.connect()
    assert result.success


def test_upload_mission_after_connect():
    bridge = _bridge()
    bridge.connect()
    result = bridge.upload_mission(_plan())
    assert result.success


def test_start_mission_without_upload_fails():
    bridge = _bridge()
    bridge.connect()
    result = bridge.start_mission()
    assert not result.success


def test_start_mission_with_upload():
    bridge = _bridge()
    bridge.connect()
    bridge.upload_mission(_plan())
    result = bridge.start_mission()
    assert result.success


def test_get_telemetry():
    bridge = _bridge()
    bridge.connect()
    t = bridge.get_telemetry()
    assert t.connected is True
    assert t.vehicle_id == "uav_001"


def test_bridge_publishes_events():
    bus = EventBus()
    received = []
    bus.subscribe("*", received.append)
    bridge = _bridge(event_bus=bus)
    bridge.connect()
    assert any(e.event_type == "mavlink_command_requested" for e in received)
    assert any(e.event_type == "mavlink_command_succeeded" for e in received)


def test_goto_updates_position():
    bridge = _bridge()
    bridge.connect()
    target = GeoPoint(lat=38.5, lon=-7.5, alt_m=50)
    bridge.goto(target)
    t = bridge.get_telemetry()
    assert t.position is not None
    assert t.position.lat == 38.5
    assert t.position.lon == -7.5
