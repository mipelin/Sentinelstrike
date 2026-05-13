"""Evidence summaries — structured extracts from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.common.time import utc_now_iso


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


def build_metrics_summary(run_dir: Path) -> dict:
    """Aggregate all per-module metrics JSONs into a single dict."""
    result: dict = {}
    for name in (
        "perception_metrics",
        "tracker_metrics",
        "geolocalizer_metrics",
        "tak_metrics",
        "realtime_metrics",
        "safety_metrics",
    ):
        data = _read_json(run_dir / f"{name}.json")
        if data is not None:
            result[name] = data
    return result


def build_operator_summary(run_dir: Path) -> dict:
    """Summarize operator decisions from the JSONL."""
    decisions = _read_jsonl(run_dir / "operator_decisions.jsonl")
    if not decisions:
        return {"total_decisions": 0}

    actions: dict[str, int] = {}
    for d in decisions:
        action = d.get("action", "unknown")
        actions[action] = actions.get(action, 0) + 1

    return {
        "total_decisions": len(decisions),
        "action_counts": actions,
        "operator_ids": list({d.get("operator_id", "unknown") for d in decisions}),
    }


def build_safety_summary(run_dir: Path) -> dict:
    """Summarize safety actions from the JSONL."""
    actions = _read_jsonl(run_dir / "safety_actions.jsonl")
    if not actions:
        return {"total_actions": 0}

    triggers: dict[str, int] = {}
    executed: dict[str, int] = {}
    for a in actions:
        t = a.get("trigger", "unknown")
        triggers[t] = triggers.get(t, 0) + 1
        e = a.get("action", "unknown")
        executed[e] = executed.get(e, 0) + 1

    return {
        "total_actions": len(actions),
        "trigger_counts": triggers,
        "action_counts": executed,
        "all_successful": all(a.get("success", True) for a in actions),
    }


def build_run_summary(run_dir: Path) -> dict:
    """Top-level run summary combining metadata, scenario info, and counts."""
    meta = _read_json(run_dir / "metadata.json") or {}
    scenario = _read_json(run_dir / "scenario_metadata.json")
    pipeline_summary = _read_json(run_dir / "pipeline_summary.json")

    counts: dict[str, int] = {}
    for suffix in (".jsonl",):
        for p in run_dir.iterdir():
            if p.is_file() and p.suffix == suffix:
                lines = p.read_text(encoding="utf-8").strip().splitlines()
                counts[p.name] = len([l for l in lines if l.strip()])

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "generated_at": utc_now_iso(),
        "metadata": meta,
        "scenario": scenario,
        "pipeline_summary": pipeline_summary,
        "artifact_line_counts": counts,
    }
