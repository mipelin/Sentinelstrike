"""Tests for MavlinkTelemetryProvider age_ms and link_ok."""

from __future__ import annotations

from sentinel.realtime.telemetry_stream import MavlinkTelemetryProvider, StaticTelemetryProvider


class _MockBridge:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.call_count = 0

    def get_telemetry(self) -> object:
        self.call_count += 1
        if self.fail:
            raise RuntimeError("simulated telemetry failure")
        from sentinel.mavlink_bridge.telemetry import TelemetrySnapshot
        return TelemetrySnapshot(
            vehicle_id="uav_001",
            timestamp_utc="2026-01-01T00:00:00Z",
            connected=True,
            armed=False,
            mode="AUTO",
            heading_deg=0.0,
            groundspeed_mps=5.0,
            battery_pct=95.0,
        )


def test_telemetry_provider_initial_link_ok_is_false() -> None:
    provider = MavlinkTelemetryProvider(mavlink_bridge=_MockBridge())
    assert provider.link_ok is False


def test_telemetry_provider_age_ms_zero_on_success() -> None:
    bridge = _MockBridge()
    provider = MavlinkTelemetryProvider(mavlink_bridge=bridge)
    state = provider.get_vehicle_state()
    assert state is not None
    assert provider.link_ok is True
    assert provider.telemetry_age_ms == 0.0


def test_telemetry_provider_stale_count_increments_on_failure() -> None:
    bridge = _MockBridge()
    provider = MavlinkTelemetryProvider(mavlink_bridge=bridge)
    provider.get_vehicle_state()
    bridge.fail = True
    result = provider.get_vehicle_state()
    assert result is not None
    assert provider.link_ok is False
    assert provider.stale_count >= 1
    assert provider.telemetry_age_ms > 0


def test_telemetry_provider_returns_last_valid_on_failure() -> None:
    bridge = _MockBridge()
    provider = MavlinkTelemetryProvider(mavlink_bridge=bridge)
    first = provider.get_vehicle_state()
    assert first is not None

    bridge.fail = True
    second = provider.get_vehicle_state()
    assert second is not None
    assert second.mode == first.mode


def test_telemetry_provider_stale_count_resets_on_recovery() -> None:
    bridge = _MockBridge()
    provider = MavlinkTelemetryProvider(mavlink_bridge=bridge)
    provider.get_vehicle_state()

    bridge.fail = True
    provider.get_vehicle_state()
    assert provider.stale_count >= 1

    bridge.fail = False
    provider.get_vehicle_state()
    assert provider.stale_count == 0
    assert provider.link_ok is True


def test_static_telemetry_provider_always_returns() -> None:
    provider = StaticTelemetryProvider()
    state = provider.get_vehicle_state()
    assert state is not None
    assert state.vehicle_id == "uav_001"


def test_telemetry_provider_with_timeout_param() -> None:
    provider = MavlinkTelemetryProvider(mavlink_bridge=_MockBridge(), telemetry_timeout_s=1.0)
    state = provider.get_vehicle_state()
    assert state is not None
    assert provider.link_ok is True
