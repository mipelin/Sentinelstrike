"""Maps domain models to CoT XML messages."""

from __future__ import annotations

from sentinel.common.events import SystemEvent
from sentinel.common.types import GeoObservation, MissionPlan, VehicleState

from .cot import build_cot_event


def vehicle_state_to_cot(
    state: VehicleState,
    callsign: str,
    stale_after_s: int = 120,
) -> str | None:
    if state.position is None:
        return None
    remarks = f"mode={state.mode}; armed={state.armed}"
    if state.battery_pct is not None:
        remarks += f"; battery={state.battery_pct:.0f}%"
    return build_cot_event(
        uid=f"ONS-UAV-{state.vehicle_id}",
        type_="a-f-A-M-F-Q",
        lat=state.position.lat,
        lon=state.position.lon,
        hae_m=state.position.alt_m or 0.0,
        ce_m=20.0,
        le_m=50.0,
        callsign=callsign,
        remarks=remarks,
        stale_after_s=stale_after_s,
    )


def geo_observation_to_cot(
    obs: GeoObservation,
    callsign_prefix: str,
    stale_after_s: int = 120,
) -> str:
    cs = f"{callsign_prefix}-{obs.class_name}-{obs.track_id}"
    remarks = (
        f"Observed {obs.class_name}; "
        f"confidence={obs.confidence:.2f}; "
        f"accuracy~{obs.accuracy_estimate_m:.1f}m; "
        f"method={obs.method}"
    )
    return build_cot_event(
        uid=f"ONS-OBS-{obs.observation_id}",
        type_="a-u-G",
        lat=obs.estimated_location.lat,
        lon=obs.estimated_location.lon,
        hae_m=obs.estimated_location.alt_m or 0.0,
        ce_m=obs.accuracy_estimate_m,
        le_m=obs.accuracy_estimate_m,
        callsign=cs,
        remarks=remarks,
        stale_after_s=stale_after_s,
    )


def mission_plan_waypoints_to_cot(
    plan: MissionPlan,
    callsign_prefix: str,
    stale_after_s: int = 120,
) -> list[str]:
    messages = []
    count = len(plan.waypoints)
    for i, wp in enumerate(plan.waypoints):
        messages.append(
            build_cot_event(
                uid=f"ONS-WP-{plan.mission_id}-{i}",
                type_="b-m-p-w",
                lat=wp.lat,
                lon=wp.lon,
                hae_m=wp.alt_m or 0.0,
                ce_m=20.0,
                le_m=50.0,
                callsign=f"{callsign_prefix}-WP-{i}",
                remarks=f"Mission waypoint {i}/{count}",
                stale_after_s=stale_after_s,
            )
        )
    return messages


def system_event_to_cot(
    event: SystemEvent,
    lat: float,
    lon: float,
    callsign_prefix: str,
    stale_after_s: int = 120,
) -> str:
    remarks = f"{event.event_type}; severity={event.severity}"
    return build_cot_event(
        uid=f"ONS-EVT-{event.event_id}",
        type_="b-a-o-tbl",
        lat=lat,
        lon=lon,
        callsign=f"{callsign_prefix}-ALERT",
        remarks=remarks,
        stale_after_s=stale_after_s,
    )
