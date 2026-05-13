"""Map layer builder — extracts geographic data for Leaflet rendering."""

from __future__ import annotations

from .models import MapLayers, ObservationLayer, SafetyEventLayer, UavPositionLayer, UavTrackPoint, WaypointLayer


def build_map_layers(artifacts: dict) -> MapLayers:
    """Build map layers from run artifacts."""
    waypoints = _extract_waypoints(artifacts)
    uav_positions, uav_track = _extract_uav_data(artifacts)
    observations = _extract_observations(artifacts)
    safety_events = _extract_safety_events(artifacts)

    if not uav_positions and waypoints:
        wp = waypoints[0]
        uav_positions = [UavPositionLayer(lat=wp.lat, lon=wp.lon, label="Launch")]

    return MapLayers(
        waypoints=waypoints,
        uav_positions=uav_positions,
        uav_track=uav_track,
        observations=observations,
        safety_events=safety_events,
    )


def _extract_waypoints(artifacts: dict) -> list[WaypointLayer]:
    plan = artifacts.get("mission_plan")
    if not plan or not isinstance(plan, dict):
        return []
    wps = plan.get("waypoints", [])
    return [
        WaypointLayer(lat=wp.get("lat", 0), lon=wp.get("lon", 0), label=f"WP {i + 1}")
        for i, wp in enumerate(wps)
        if isinstance(wp, dict) and "lat" in wp and "lon" in wp
    ]


def _extract_uav_data(artifacts: dict) -> tuple[list[UavPositionLayer], list[UavTrackPoint]]:
    """Extract UAV positions and track from vehicle_states.jsonl."""
    positions: list[UavPositionLayer] = []
    track: list[UavTrackPoint] = []

    for vs in artifacts.get("vehicle_states", []):
        pos = vs.get("position")
        if not isinstance(pos, dict) or "lat" not in pos or "lon" not in pos:
            continue
        positions.append(
            UavPositionLayer(
                lat=pos["lat"],
                lon=pos["lon"],
                timestamp=vs.get("timestamp_utc", ""),
                frame_id=vs.get("frame_id"),
                heading=vs.get("heading_deg"),
                alt_m=pos.get("alt_m"),
                speed_mps=vs.get("groundspeed_mps"),
                battery_pct=vs.get("battery_pct"),
                mode=vs.get("mode", ""),
                armed=vs.get("armed", False),
                stale=vs.get("stale", False),
                vehicle_id=vs.get("vehicle_id", ""),
            )
        )
        track.append(
            UavTrackPoint(
                lat=pos["lat"],
                lon=pos["lon"],
                alt_m=pos.get("alt_m"),
                timestamp_utc=vs.get("timestamp_utc", ""),
                frame_id=vs.get("frame_id"),
            )
        )

    if positions:
        return positions, track

    # Fallback to events
    for ev in artifacts.get("events", []):
        payload = ev.get("payload", {})
        pos = payload.get("position")
        if isinstance(pos, dict) and "lat" in pos and "lon" in pos:
            positions.append(
                UavPositionLayer(
                    lat=pos["lat"],
                    lon=pos["lon"],
                    timestamp=ev.get("timestamp_utc", ""),
                    heading=payload.get("heading_deg"),
                )
            )
    if positions:
        return positions, []

    metadata = artifacts.get("metadata")
    if isinstance(metadata, dict):
        pos = metadata.get("vehicle_position")
        if isinstance(pos, dict) and "lat" in pos and "lon" in pos:
            return [UavPositionLayer(lat=pos["lat"], lon=pos["lon"])], []

    return [], []


def _extract_observations(artifacts: dict) -> list[ObservationLayer]:
    confirmed_ids = _build_confirmed_ids(artifacts)
    obs_list: list[ObservationLayer] = []
    for obs in artifacts.get("geo_observations", []):
        loc = obs.get("estimated_location", {})
        if not isinstance(loc, dict) or "lat" not in loc or "lon" not in loc:
            continue
        obs_id = obs.get("observation_id", "")
        obs_list.append(
            ObservationLayer(
                lat=loc["lat"],
                lon=loc["lon"],
                track_id=obs.get("track_id", ""),
                class_name=obs.get("class_name", ""),
                confidence=obs.get("confidence", 0),
                confirmed=obs_id in confirmed_ids,
                observation_id=obs_id,
                frame_id=(obs.get("metadata", {}) or {}).get("frame_id"),
            )
        )
    return obs_list


def _build_confirmed_ids(artifacts: dict) -> set[str]:
    ids: set[str] = set()
    for dec in artifacts.get("operator_decisions", []):
        if dec.get("action") == "CONFIRM_OBSERVATION":
            oid = dec.get("observation_id")
            if oid:
                ids.add(oid)
    return ids


def _extract_safety_events(artifacts: dict) -> list[SafetyEventLayer]:
    events: list[SafetyEventLayer] = []
    for sa in artifacts.get("safety_actions", []):
        events.append(
            SafetyEventLayer(
                lat=None,
                lon=None,
                action=sa.get("action", ""),
                trigger=sa.get("trigger", ""),
            )
        )
    return events
