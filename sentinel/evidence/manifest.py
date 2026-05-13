"""Replay manifest — artifact inventory with metadata for deterministic replay."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.common.time import utc_now_iso

from .checksums import compute_checksums

REPLAY_VERSION = "1.0"


def build_replay_manifest(run_dir: Path) -> dict:
    """Build replay_manifest.json for the run directory."""
    meta = _read_json(run_dir / "metadata.json") or {}
    scenario = _read_json(run_dir / "scenario_metadata.json")

    artifact_inventory: list[dict] = []
    checksums = compute_checksums(run_dir)

    for p in sorted(run_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            entry: dict = {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": checksums.get(p.name, ""),
            }
            artifact_inventory.append(entry)

    return {
        "replay_version": REPLAY_VERSION,
        "run_id": run_dir.name,
        "generated_at": utc_now_iso(),
        "mission_id": meta.get("mission_id", ""),
        "scenario": scenario,
        "artifact_count": len(artifact_inventory),
        "total_size_bytes": sum(a["size_bytes"] for a in artifact_inventory),
        "artifacts": artifact_inventory,
    }


def build_artifacts_index(run_dir: Path) -> dict:
    """Build artifacts_index.json — human-readable index of all artifacts by category."""
    index: dict[str, list[dict]] = {
        "events": [],
        "perception": [],
        "tracking": [],
        "geolocalization": [],
        "tak": [],
        "operator": [],
        "safety": [],
        "mission": [],
        "report": [],
        "other": [],
    }

    category_map = {
        "events.jsonl": "events",
        "detections.jsonl": "perception",
        "perception_metrics.json": "perception",
        "tracks.jsonl": "tracking",
        "tracker_metrics.json": "tracking",
        "geo_observations.jsonl": "geolocalization",
        "geolocalizer_metrics.json": "geolocalization",
        "tak_messages.jsonl": "tak",
        "tak_metrics.json": "tak",
        "operator_decisions.jsonl": "operator",
        "safety_actions.jsonl": "safety",
        "safety_metrics.json": "safety",
        "mission_plan.json": "mission",
        "metadata.json": "mission",
        "scenario_metadata.json": "mission",
        "pipeline_summary.json": "mission",
        "realtime_metrics.json": "mission",
        "report.md": "report",
    }

    for p in sorted(run_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            cat = category_map.get(p.name, "other")
            index[cat].append({
                "filename": p.name,
                "size_bytes": p.stat().st_size,
            })

    # Remove empty categories
    return {k: v for k, v in index.items() if v}


def build_consolidated_timeline(run_dir: Path) -> list[dict]:
    """Merge all event-like JSONL files into a single chronological timeline."""
    timeline: list[dict] = []

    sources = [
        ("events.jsonl", "system_event"),
        ("operator_decisions.jsonl", "operator_decision"),
        ("safety_actions.jsonl", "safety_action"),
        ("geo_observations.jsonl", "geo_observation"),
    ]

    for filename, source_type in sources:
        for item in _read_jsonl(run_dir / filename):
            item["_source_type"] = source_type
            timeline.append(item)

    timeline.sort(key=lambda e: e.get("timestamp_utc", ""))
    return timeline


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items
