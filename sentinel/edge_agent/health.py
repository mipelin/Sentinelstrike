"""Edge agent health snapshot builder."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


def _artifact_exists(run_dir: Path | None, filename: str) -> bool:
    if run_dir is None:
        return False
    return (run_dir / filename).exists()


def build_health_snapshot(context: EdgeAgentContext) -> dict:  # type: ignore[name-defined]  # noqa: F821
    vehicle_info: dict = {}
    if context.vehicle_state is not None:
        vs = context.vehicle_state
        vehicle_info = {
            "connected": True,
            "armed": vs.armed,
            "mode": vs.mode,
            "battery_pct": vs.battery_pct,
            "position": {
                "lat": vs.position.lat,
                "lon": vs.position.lon,
                "alt_m": vs.position.alt_m,
            }
            if vs.position
            else None,
        }

    run_dir = context.run_dir
    return {
        "mode": context.mode,
        "mission_id": context.mission_request.mission_id,
        "run_dir": str(run_dir) if run_dir else None,
        "has_recorder": context.recorder is not None,
        "has_mavlink": context.mavlink_bridge is not None,
        "has_tak": context.tak_bridge is not None,
        "vehicle": vehicle_info or None,
        "artifacts": {
            "events_jsonl": _artifact_exists(run_dir, "events.jsonl"),
            "mission_plan_json": _artifact_exists(run_dir, "mission_plan.json"),
            "detections_jsonl": _artifact_exists(run_dir, "detections.jsonl"),
            "tracks_jsonl": _artifact_exists(run_dir, "tracks.jsonl"),
            "geo_observations_jsonl": _artifact_exists(run_dir, "geo_observations.jsonl"),
            "tak_messages_jsonl": _artifact_exists(run_dir, "tak_messages.jsonl"),
            "pipeline_summary_json": _artifact_exists(run_dir, "pipeline_summary.json"),
            "report_md": _artifact_exists(run_dir, "report.md"),
            "edge_agent_health_json": _artifact_exists(run_dir, "edge_agent_health.json"),
        },
    }


def write_health_snapshot(context: EdgeAgentContext, output_path: Path) -> None:  # type: ignore[name-defined]  # noqa: F821
    snapshot = build_health_snapshot(context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info("Health snapshot written to {}", output_path)
