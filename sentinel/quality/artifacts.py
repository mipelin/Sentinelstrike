"""Pipeline artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

from .jsonl import validate_jsonl_files

EXPECTED_PIPELINE_ARTIFACTS = [
    "metadata.json",
    "events.jsonl",
    "mission_plan.json",
    "detections.jsonl",
    "tracks.jsonl",
    "geo_observations.jsonl",
    "tak_messages.jsonl",
    "perception_metrics.json",
    "tracker_metrics.json",
    "geolocalizer_metrics.json",
    "tak_metrics.json",
    "pipeline_summary.json",
    "report.md",
]


def validate_pipeline_run_dir(run_dir: Path) -> dict:
    errors: list[str] = []
    missing: list[str] = []
    json_errors: list[str] = []

    if not run_dir.exists():
        return {
            "run_dir": str(run_dir),
            "valid": False,
            "missing": EXPECTED_PIPELINE_ARTIFACTS[:],
            "json_errors": [],
            "jsonl_results": [],
            "report_exists": False,
            "summary_exists": False,
            "errors": ["run_dir does not exist"],
        }

    for name in EXPECTED_PIPELINE_ARTIFACTS:
        p = run_dir / name
        if not p.exists():
            missing.append(name)

    # Validate JSON files
    for name in EXPECTED_PIPELINE_ARTIFACTS:
        if not name.endswith(".json"):
            continue
        p = run_dir / name
        if not p.exists():
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception) as exc:
            json_errors.append(f"{name}: {exc}")

    # Validate JSONL files
    jsonl_names = [n for n in EXPECTED_PIPELINE_ARTIFACTS if n.endswith(".jsonl")]
    jsonl_paths = [run_dir / n for n in jsonl_names]
    jsonl_results = validate_jsonl_files(jsonl_paths)
    for r in jsonl_results:
        if r["exists"] and not r["valid"]:
            json_errors.append(r["error"])

    report_exists = (run_dir / "report.md").exists()
    summary_exists = (run_dir / "pipeline_summary.json").exists()

    # Validate summary structure
    if summary_exists:
        try:
            summary = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            for key in ("mission_id", "planner", "artifacts"):
                if key not in summary:
                    errors.append(f"pipeline_summary.json missing key: {key}")
        except Exception as exc:
            errors.append(f"pipeline_summary.json: {exc}")

    report_nonempty = False
    if report_exists:
        report_nonempty = (run_dir / "report.md").stat().st_size > 0
        if not report_nonempty:
            errors.append("report.md is empty")

    valid = not missing and not json_errors and not errors and report_exists and summary_exists

    return {
        "run_dir": str(run_dir),
        "valid": valid,
        "missing": missing,
        "json_errors": json_errors,
        "jsonl_results": jsonl_results,
        "report_exists": report_exists,
        "summary_exists": summary_exists,
        "errors": errors,
    }


def find_latest_run_dir(base_dir: Path = Path("runs")) -> Path | None:
    if not base_dir.exists():
        return None
    dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[0] if dirs else None
