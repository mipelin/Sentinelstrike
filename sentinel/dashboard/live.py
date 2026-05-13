"""Live run session — tailing JSONL artifacts for real-time dashboard updates."""

from __future__ import annotations

from pathlib import Path

from sentinel.common.time import utc_now_iso

from .map_layers import build_map_layers
from .state import build_operator_queue, build_target_cards
from .tailer import JsonlTailer
from .timeline import build_timeline

_TAILER_FILES = [
    "events",
    "detections",
    "tracks",
    "geo_observations",
    "operator_decisions",
    "safety_actions",
    "tak_messages",
    "vehicle_states",
]


class LiveRunSession:
    """Manages incremental tailing of all JSONL artifacts for a single run."""

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self._tailers: dict[str, JsonlTailer] = {}
        for name in _TAILER_FILES:
            self._tailers[name] = JsonlTailer(run_dir / f"{name}.jsonl")

        # Accumulated data for map/timeline building
        self._all_events: list[dict] = []
        self._all_geo_observations: list[dict] = []
        self._all_operator_decisions: list[dict] = []
        self._all_safety_actions: list[dict] = []
        self._all_tracks: list[dict] = []
        self._all_vehicle_states: list[dict] = []

    def poll(self) -> dict:
        """Poll all tailers and return updated dashboard payload."""
        new_events = self._tailers["events"].poll()
        self._all_events.extend(new_events)

        new_obs = self._tailers["geo_observations"].poll()
        self._all_geo_observations.extend(new_obs)

        new_ops = self._tailers["operator_decisions"].poll()
        self._all_operator_decisions.extend(new_ops)

        new_safety = self._tailers["safety_actions"].poll()
        self._all_safety_actions.extend(new_safety)

        new_tracks = self._tailers["tracks"].poll()
        self._all_tracks.extend(new_tracks)

        new_vs = self._tailers["vehicle_states"].poll()
        self._all_vehicle_states.extend(new_vs)

        # Poll the rest for counts only
        self._tailers["detections"].poll()
        self._tailers["tak_messages"].poll()

        counts = {name: t.line_count for name, t in self._tailers.items()}

        artifacts = {
            "events": self._all_events,
            "geo_observations": self._all_geo_observations,
            "operator_decisions": self._all_operator_decisions,
            "safety_actions": self._all_safety_actions,
            "tracks": self._all_tracks,
            "vehicle_states": self._all_vehicle_states,
        }

        timeline = build_timeline(artifacts)
        tail = [e.model_dump() for e in timeline[-20:]]

        map_data = build_map_layers(artifacts)
        targets = build_target_cards(artifacts)
        operator_queue = build_operator_queue(artifacts)

        # Extract latest UAV state for telemetry panel
        latest_uav = None
        if self._all_vehicle_states:
            last = self._all_vehicle_states[-1]
            pos = last.get("position")
            if isinstance(pos, dict):
                latest_uav = {
                    "vehicle_id": last.get("vehicle_id", ""),
                    "lat": pos.get("lat"),
                    "lon": pos.get("lon"),
                    "alt_m": pos.get("alt_m"),
                    "heading_deg": last.get("heading_deg"),
                    "groundspeed_mps": last.get("groundspeed_mps"),
                    "battery_pct": last.get("battery_pct"),
                    "mode": last.get("mode", ""),
                    "armed": last.get("armed", False),
                    "stale": last.get("stale", False),
                    "timestamp_utc": last.get("timestamp_utc", ""),
                }

        # Track tail (last 200 points)
        uav_track_tail = []
        for vs in self._all_vehicle_states[-200:]:
            pos = vs.get("position")
            if isinstance(pos, dict) and "lat" in pos and "lon" in pos:
                uav_track_tail.append({
                    "lat": pos["lat"],
                    "lon": pos["lon"],
                    "alt_m": pos.get("alt_m"),
                    "timestamp_utc": vs.get("timestamp_utc", ""),
                })

        latest_frame_name = ""
        frames_dir = self.run_dir / "frames"
        if frames_dir.exists():
            frame_names = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
            if frame_names:
                latest_frame_name = frame_names[-1]

        return {
            "run_id": self.run_id,
            "timestamp_utc": utc_now_iso(),
            "artifact_counts": counts,
            "timeline_tail": tail,
            "map_layers": map_data.model_dump(),
            "targets": [target.model_dump() for target in targets],
            "operator_queue": [item.model_dump() for item in operator_queue],
            "latest_uav_state": latest_uav,
            "uav_track_tail": uav_track_tail,
            "latest_frame_name": latest_frame_name,
            "latest_frame_url": f"/api/runs/{self.run_id}/frames/latest.jpg?t={counts['vehicle_states']}",
        }

    def reset(self) -> None:
        """Reset all tailers to beginning."""
        for t in self._tailers.values():
            t.reset()
        self._all_events.clear()
        self._all_geo_observations.clear()
        self._all_operator_decisions.clear()
        self._all_safety_actions.clear()
        self._all_tracks.clear()
        self._all_vehicle_states.clear()
