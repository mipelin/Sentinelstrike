"""Tests for demo scenario catalog."""

from sentinel.demo.catalog import get_scenario, list_scenarios, scenario_ids


def test_list_scenarios_returns_all():
    scenarios = list_scenarios()
    assert len(scenarios) == 5
    ids = {s.scenario_id for s in scenarios}
    assert ids == {"observation_confirmed", "operator_abort", "low_battery_return", "link_loss_return", "px4_sitl_live_observation"}


def test_get_scenario_existing():
    s = get_scenario("observation_confirmed")
    assert s is not None
    assert s.name == "Confirmed Observation"
    assert s.operator_mode == "confirm_above_threshold"
    assert s.requires_px4 is False


def test_get_scenario_missing():
    assert get_scenario("nonexistent") is None


def test_scenario_ids():
    ids = scenario_ids()
    assert "observation_confirmed" in ids
    assert len(ids) == 5


def test_abort_scenario_has_safety_trigger():
    s = get_scenario("operator_abort")
    assert s.safety_trigger == "abort"


def test_low_battery_has_battery():
    s = get_scenario("low_battery_return")
    assert s.simulated_battery_pct == 10.0
    assert s.safety_trigger == "low_battery"


def test_link_loss_scenario():
    s = get_scenario("link_loss_return")
    assert s.simulated_link_ok is False
    assert s.safety_trigger == "link_loss"


def test_px4_scenario_requires_px4():
    s = get_scenario("px4_sitl_live_observation")
    assert s.requires_px4 is True
    assert s.mode == "px4_sitl"
