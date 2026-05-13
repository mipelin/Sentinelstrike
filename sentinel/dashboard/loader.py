"""Run artifact loader — reads JSON/JSONL from run directories."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .models import RunInfo


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def list_run_dirs(base_dir: Path = Path("runs")) -> list[RunInfo]:
    if not base_dir.exists():
        return []
    runs: list[RunInfo] = []
    for d in sorted(base_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        metadata = read_json(d / "metadata.json") or {}
        runs.append(
            RunInfo(
                run_id=d.name,
                path=str(d),
                created_at=_format_created_at(d.name),
                mission_id=metadata.get("mission_id", ""),
                has_report=(d / "report.md").exists(),
                has_summary=(d / "pipeline_summary.json").exists() or (d / "realtime_metrics.json").exists(),
            )
        )
    return runs


def load_run_artifacts(run_dir: Path) -> dict:
    """Load all artifacts from a run directory into a dict."""
    frames_dir = run_dir / "frames"
    return {
        "metadata": read_json(run_dir / "metadata.json"),
        "mission_plan": read_json(run_dir / "mission_plan.json"),
        "pipeline_summary": read_json(run_dir / "pipeline_summary.json"),
        "realtime_metrics": read_json(run_dir / "realtime_metrics.json"),
        "perception_metrics": read_json(run_dir / "perception_metrics.json"),
        "tracker_metrics": read_json(run_dir / "tracker_metrics.json"),
        "geolocalizer_metrics": read_json(run_dir / "geolocalizer_metrics.json"),
        "tak_metrics": read_json(run_dir / "tak_metrics.json"),
        "operator_metrics": read_json(run_dir / "operator_metrics.json"),
        "safety_metrics": read_json(run_dir / "safety_metrics.json"),
        "events": read_jsonl(run_dir / "events.jsonl"),
        "detections": read_jsonl(run_dir / "detections.jsonl"),
        "tracks": read_jsonl(run_dir / "tracks.jsonl"),
        "geo_observations": read_jsonl(run_dir / "geo_observations.jsonl"),
        "tak_messages": read_jsonl(run_dir / "tak_messages.jsonl"),
        "operator_decisions": read_jsonl(run_dir / "operator_decisions.jsonl"),
        "safety_actions": read_jsonl(run_dir / "safety_actions.jsonl"),
        "vehicle_states": read_jsonl(run_dir / "vehicle_states.jsonl"),
        "report_md": _read_text(run_dir / "report.md"),
        "frames": sorted(p.name for p in frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else [],
    }


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _format_created_at(run_id: str) -> str:
    try:
        dt = datetime.strptime(run_id[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return dt.isoformat()
    except ValueError:
        return ""
