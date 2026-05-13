"""Tests for battery percentage normalization."""


from sentinel.common.time import utc_now_iso
from sentinel.common.types import VehicleState
from sentinel.mavlink_bridge.telemetry import TelemetrySnapshot, normalize_battery_pct

# ---------------------------------------------------------------------------
# normalize_battery_pct pure function tests
# ---------------------------------------------------------------------------


def test_none_returns_none():
    assert normalize_battery_pct(None) is None


def test_fractional_scaled_to_percent():
    assert normalize_battery_pct(0.89) == 89.0


def test_fractional_zero():
    assert normalize_battery_pct(0.0) == 0.0


def test_fractional_one():
    assert normalize_battery_pct(1.0) == 100.0


def test_already_percent():
    assert normalize_battery_pct(89.0) == 89.0


def test_already_percent_100():
    assert normalize_battery_pct(100.0) == 100.0


def test_double_scaled_8900():
    """PX4 SITL bug: remaining_percent was already 0..100, *100 gives 8900."""
    assert normalize_battery_pct(8900.0) == 89.0


def test_double_scaled_5000():
    assert normalize_battery_pct(5000.0) == 50.0


def test_negative_returns_none():
    assert normalize_battery_pct(-1.0) is None


def test_nan_returns_none():
    assert normalize_battery_pct(float("nan")) is None


def test_inf_returns_none():
    assert normalize_battery_pct(float("inf")) is None


def test_very_large_clamps_to_100():
    """Values > 10000 that can't be rescued by /100 get clamped to 100."""
    assert normalize_battery_pct(15000.0) == 100.0


def test_rounds_to_2_decimals():
    assert normalize_battery_pct(0.8956) == 89.56


# ---------------------------------------------------------------------------
# TelemetrySnapshot.to_vehicle_state normalizes battery
# ---------------------------------------------------------------------------


def test_snapshot_to_vehicle_state_normalizes_8900():
    snap = TelemetrySnapshot(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        connected=True,
        armed=True,
        mode="OFFBOARD",
        battery_pct=8900.0,
    )
    vs = snap.to_vehicle_state()
    assert vs.battery_pct == 89.0


def test_snapshot_to_vehicle_state_normalizes_fraction():
    snap = TelemetrySnapshot(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        battery_pct=0.75,
    )
    vs = snap.to_vehicle_state()
    assert vs.battery_pct == 75.0


def test_snapshot_to_vehicle_state_none_battery():
    snap = TelemetrySnapshot(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        battery_pct=None,
    )
    vs = snap.to_vehicle_state()
    assert vs.battery_pct is None


def test_vehicle_state_rejects_over_100_without_normalization():
    """VehicleState itself validates battery_pct <= 100, proving the fix is needed."""
    import pytest

    with pytest.raises(Exception):
        VehicleState(
            vehicle_id="uav_001",
            timestamp_utc=utc_now_iso(),
            battery_pct=8900.0,
        )


def test_snapshot_with_normal_battery_passes():
    snap = TelemetrySnapshot(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        battery_pct=87.5,
    )
    vs = snap.to_vehicle_state()
    assert vs.battery_pct == 87.5
