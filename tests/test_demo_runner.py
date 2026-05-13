"""Tests for demo scenario runner."""

import json
from pathlib import Path

from sentinel.demo.catalog import get_scenario
from sentinel.demo.runner import run_scenario


def test_observation_confirmed_produces_artifacts():
    s = get_scenario("observation_confirmed")
    assert s is not None
    summary = run_scenario(s)

    assert summary["scenario_id"] == "observation_confirmed"
    assert "run_dir" in summary
    assert summary["metrics"]["frame_count"] > 0

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "scenario_metadata.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "realtime_metrics.json").exists()


def test_observation_confirmed_has_operator_decisions():
    s = get_scenario("observation_confirmed")
    summary = run_scenario(s)
    run_dir = Path(summary["run_dir"])

    meta = json.loads((run_dir / "scenario_metadata.json").read_text())
    assert meta["scenario_id"] == "observation_confirmed"


def test_operator_abort_produces_safety_actions():
    s = get_scenario("operator_abort")
    summary = run_scenario(s)

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "safety_actions.jsonl").exists()
    lines = (run_dir / "safety_actions.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 2  # HOLD + LAND


def test_low_battery_produces_safety_actions():
    s = get_scenario("low_battery_return")
    summary = run_scenario(s)

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "safety_actions.jsonl").exists()
    lines = (run_dir / "safety_actions.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 1
    action = json.loads(lines[0])
    assert action["trigger"] == "LOW_BATTERY"


def test_link_loss_produces_safety_actions():
    s = get_scenario("link_loss_return")
    summary = run_scenario(s)

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "safety_actions.jsonl").exists()
    lines = (run_dir / "safety_actions.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 1
    action = json.loads(lines[0])
    assert action["trigger"] == "LINK_LOSS"


def test_scenario_metadata_has_all_fields():
    s = get_scenario("observation_confirmed")
    summary = run_scenario(s)

    assert "scenario_id" in summary
    assert "scenario_name" in summary
    assert "run_id" in summary
    assert "executed_at" in summary
    assert "metrics" in summary


def test_scenario_run_dir_named_with_scenario():
    s = get_scenario("observation_confirmed")
    summary = run_scenario(s)
    assert "observation_confirmed" in summary["run_id"]


def test_evidence_package_created_after_scenario():
    s = get_scenario("observation_confirmed")
    summary = run_scenario(s)
    run_dir = Path(summary["run_dir"])

    pkg_dir = run_dir / "mission_package"
    assert pkg_dir.exists()
    assert (pkg_dir / "summary.json").exists()
    assert (pkg_dir / "replay_manifest.json").exists()
    assert (pkg_dir / "timeline.json").exists()
    assert (pkg_dir / "checksums.json").exists()

    import json
    manifest = json.loads((pkg_dir / "replay_manifest.json").read_text())
    assert manifest["run_id"] == run_dir.name
