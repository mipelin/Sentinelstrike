"""Pipeline summary builder."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.common.types import MissionPlan


def build_pipeline_summary(
    mission_id: str,
    run_dir: Path,
    mission_plan: MissionPlan,
    perception_metrics: dict | None = None,
    tracker_metrics: dict | None = None,
    geolocalizer_metrics: dict | None = None,
    tak_metrics: dict | None = None,
    mavlink_results: list[dict] | None = None,
) -> dict:
    total_cmds = len(mavlink_results) if mavlink_results else 0
    success_cmds = sum(1 for r in (mavlink_results or []) if r.get("success"))
    failed_cmds = total_cmds - success_cmds

    return {
        "mission_id": mission_id,
        "run_dir": str(run_dir),
        "planner": {
            "version": mission_plan.planner_version,
            "waypoint_count": len(mission_plan.waypoints),
            "total_distance_m": mission_plan.total_distance_m,
            "estimated_duration_s": mission_plan.estimated_duration_s,
        },
        "perception": {
            "frame_count": (perception_metrics or {}).get("frame_count", 0),
            "detection_count": (perception_metrics or {}).get("detection_count", 0),
        },
        "tracking": {
            "total_tracks": (tracker_metrics or {}).get("total_tracks", 0),
            "active_tracks": (tracker_metrics or {}).get("active_tracks", 0),
            "lost_tracks": (tracker_metrics or {}).get("lost_tracks", 0),
            "terminated_tracks": (tracker_metrics or {}).get("terminated_tracks", 0),
        },
        "geolocalization": {
            "observation_count": (geolocalizer_metrics or {}).get("observation_count", 0),
        },
        "tak": {
            "message_count": (tak_metrics or {}).get("message_count", 0),
            "sent_count": (tak_metrics or {}).get("sent_count", 0),
            "failed_count": (tak_metrics or {}).get("failed_count", 0),
        },
        "mavlink": {
            "command_count": total_cmds,
            "success_count": success_cmds,
            "failed_count": failed_cmds,
        },
        "artifacts": {
            "events": "events.jsonl",
            "mission_plan": "mission_plan.json",
            "detections": "detections.jsonl",
            "tracks": "tracks.jsonl",
            "geo_observations": "geo_observations.jsonl",
            "tak_messages": "tak_messages.jsonl",
            "report": "report.md",
        },
    }


def write_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
