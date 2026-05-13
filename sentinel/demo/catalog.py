"""Demo scenario catalog — registry of all available scenarios."""

from __future__ import annotations

from .scenarios import DemoScenario

_SCENARIOS: dict[str, DemoScenario] = {
    "observation_confirmed": DemoScenario(
        scenario_id="observation_confirmed",
        name="Confirmed Observation",
        description="Full pipeline: detection, tracking, geolocalization, operator confirms, TAK publishes confirmed observation.",
        operator_mode="confirm_above_threshold",
    ),
    "operator_abort": DemoScenario(
        scenario_id="operator_abort",
        name="Operator Abort",
        description="Operator simulator generates ABORT_MISSION, safety executor executes HOLD + LAND.",
        operator_mode="auto_reject",
        safety_trigger="abort",
    ),
    "low_battery_return": DemoScenario(
        scenario_id="low_battery_return",
        name="Low Battery Return",
        description="Simulated low battery telemetry triggers RETURN_HOME via safety executor.",
        safety_trigger="low_battery",
        simulated_battery_pct=10.0,
    ),
    "link_loss_return": DemoScenario(
        scenario_id="link_loss_return",
        name="Link Loss Return",
        description="Simulated link loss triggers RETURN_HOME via safety executor.",
        safety_trigger="link_loss",
        simulated_link_ok=False,
    ),
    "px4_sitl_live_observation": DemoScenario(
        scenario_id="px4_sitl_live_observation",
        name="PX4 SITL Live Observation",
        description="PX4 SITL with local mission, realtime loop, dashboard live compatible. Requires PX4 running.",
        mode="px4_sitl",
        requires_px4=True,
        target_fps=5.0,
        operator_mode="auto_confirm",
    ),
}


def get_scenario(scenario_id: str) -> DemoScenario | None:
    return _SCENARIOS.get(scenario_id)


def list_scenarios() -> list[DemoScenario]:
    return list(_SCENARIOS.values())


def scenario_ids() -> list[str]:
    return list(_SCENARIOS.keys())
