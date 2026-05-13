"""Tests for EdgeAgentContext."""


import pytest

from sentinel.common.event_bus import EventBus
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoPoint, MissionRequest, VehicleState
from sentinel.config.schema import AppConfig
from sentinel.edge_agent.context import EdgeAgentContext


def _request():
    return MissionRequest(
        mission_id="ctx_test",
        launch_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=0),
        area_of_interest=[
            GeoPoint(lat=38.000, lon=-8.000, alt_m=80),
            GeoPoint(lat=38.000, lon=-7.998, alt_m=80),
            GeoPoint(lat=38.002, lon=-7.998, alt_m=80),
        ],
    )


def _context():
    return EdgeAgentContext(
        config=AppConfig(),
        mission_request=_request(),
        event_bus=EventBus(),
        mode="mock",
    )


def test_require_run_dir_fails_when_none():
    ctx = _context()
    with pytest.raises(RuntimeError, match="Run directory not initialized"):
        ctx.require_run_dir()


def test_close_all_idempotent():
    ctx = _context()
    ctx.close_all()
    ctx.close_all()
    ctx.close_all()


def test_set_vehicle_state():
    ctx = _context()
    state = VehicleState(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
        heading_deg=90.0,
        mode="SIMULATED",
        armed=False,
        battery_pct=90.0,
    )
    ctx.set_vehicle_state(state)
    assert ctx.vehicle_state is not None
    assert ctx.vehicle_state.vehicle_id == "uav_001"
    assert ctx.vehicle_state.position.lat == 38.0


def test_close_all_with_mock_subsystems():
    ctx = _context()

    class MockCloseable:
        self_closed = False

        def close(self):
            self.self_closed = True

    mock = MockCloseable()
    ctx.recorder = mock
    ctx.mavlink_bridge = mock
    ctx.tak_bridge = mock
    ctx.close_all()
    assert mock.self_closed is True
