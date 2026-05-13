"""Tests for TAK mapper."""

import xml.etree.ElementTree as ET

from sentinel.common.events import SystemEvent
from sentinel.common.time import utc_now_iso
from sentinel.common.types import (
    GeoObservation,
    GeoPoint,
    MissionPlan,
    VehicleState,
)
from sentinel.tak_bridge.cot import validate_cot_xml
from sentinel.tak_bridge.mapper import (
    geo_observation_to_cot,
    mission_plan_waypoints_to_cot,
    system_event_to_cot,
    vehicle_state_to_cot,
)


def _vehicle(lat=38.0, lon=-8.0, alt=80.0):
    return VehicleState(
        vehicle_id="uav_001",
        timestamp_utc=utc_now_iso(),
        position=GeoPoint(lat=lat, lon=lon, alt_m=alt),
        heading_deg=90.0,
        mode="SIMULATED",
        armed=False,
        battery_pct=90.0,
    )


def _obs():
    return GeoObservation(
        observation_id="obs_001",
        mission_id="m1",
        track_id="trk_000001",
        class_name="person",
        timestamp_utc=utc_now_iso(),
        estimated_location=GeoPoint(lat=38.001, lon=-8.001, alt_m=0.0),
        accuracy_estimate_m=35.0,
        confidence=0.85,
    )


def test_vehicle_state_to_cot_none_without_position():
    state = VehicleState(vehicle_id="uav_001", timestamp_utc=utc_now_iso())
    assert vehicle_state_to_cot(state, "CS") is None


def test_vehicle_state_to_cot_valid_xml():
    xml = vehicle_state_to_cot(_vehicle(), "ONS-1")
    assert xml is not None
    assert validate_cot_xml(xml)
    root = ET.fromstring(xml)
    assert "ONS-UAV-" in root.get("uid")


def test_geo_observation_includes_details():
    xml = geo_observation_to_cot(_obs(), "ONS")
    assert validate_cot_xml(xml)
    root = ET.fromstring(xml)
    remarks = root.find("detail/remarks")
    assert remarks is not None
    assert "person" in remarks.text
    assert "0.85" in remarks.text
    assert "35.0m" in remarks.text


def test_mission_plan_waypoints_count():
    plan = MissionPlan(
        mission_id="m1",
        waypoints=[
            GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
            GeoPoint(lat=38.01, lon=-8.0, alt_m=80),
            GeoPoint(lat=38.02, lon=-8.0, alt_m=80),
        ],
        search_pattern="lawnmower",
        return_home_point=GeoPoint(lat=38.0, lon=-8.0, alt_m=80),
    )
    msgs = mission_plan_waypoints_to_cot(plan, "ONS")
    assert len(msgs) == 3
    for m in msgs:
        assert validate_cot_xml(m)


def test_system_event_to_cot():
    evt = SystemEvent(
        event_id="abc123",
        mission_id="m1",
        timestamp_utc=utc_now_iso(),
        source="test",
        event_type="test_alert",
        severity="warning",
    )
    xml = system_event_to_cot(evt, 38.0, -8.0, "ONS")
    assert validate_cot_xml(xml)
    root = ET.fromstring(xml)
    assert root.get("type") == "b-a-o-tbl"
    assert "ONS-EVT-" in root.get("uid")
