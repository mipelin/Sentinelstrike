"""Tests for evidence summaries."""

import json
from pathlib import Path

import pytest

from sentinel.evidence.summaries import (
    build_metrics_summary,
    build_operator_summary,
    build_run_summary,
    build_safety_summary,
)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "20260511T120000Z_test_run"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"mission_id": "test_001", "start_time": "2026-05-11T12:00:00Z"}))
    (d / "perception_metrics.json").write_text(json.dumps({"frame_count": 30, "detection_count": 45}))
    (d / "tracker_metrics.json").write_text(json.dumps({"total_tracks": 12, "active_tracks": 5}))
    (d / "events.jsonl").write_text('{"event_type":"test"}\n{"event_type":"test2"}\n')
    return d


def test_build_metrics_summary(run_dir: Path):
    result = build_metrics_summary(run_dir)
    assert "perception_metrics" in result
    assert result["perception_metrics"]["frame_count"] == 30
    assert "tracker_metrics" in result
    assert result["tracker_metrics"]["total_tracks"] == 12


def test_build_metrics_summary_missing_files(tmp_path: Path):
    result = build_metrics_summary(tmp_path)
    assert result == {}


def test_build_operator_summary_with_decisions(run_dir: Path):
    decisions = [
        {"action": "CONFIRM", "operator_id": "op_1"},
        {"action": "CONFIRM", "operator_id": "op_1"},
        {"action": "REJECT", "operator_id": "op_2"},
    ]
    (run_dir / "operator_decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions),
    )

    result = build_operator_summary(run_dir)
    assert result["total_decisions"] == 3
    assert result["action_counts"]["CONFIRM"] == 2
    assert result["action_counts"]["REJECT"] == 1
    assert sorted(result["operator_ids"]) == ["op_1", "op_2"]


def test_build_operator_summary_empty(run_dir: Path):
    result = build_operator_summary(run_dir)
    assert result["total_decisions"] == 0


def test_build_safety_summary_with_actions(run_dir: Path):
    actions = [
        {"trigger": "LOW_BATTERY", "action": "RETURN_HOME", "success": True},
        {"trigger": "LOW_BATTERY", "action": "RETURN_HOME", "success": True},
    ]
    (run_dir / "safety_actions.jsonl").write_text(
        "\n".join(json.dumps(a) for a in actions),
    )

    result = build_safety_summary(run_dir)
    assert result["total_actions"] == 2
    assert result["trigger_counts"]["LOW_BATTERY"] == 2
    assert result["action_counts"]["RETURN_HOME"] == 2
    assert result["all_successful"] is True


def test_build_safety_summary_with_failure(run_dir: Path):
    actions = [
        {"trigger": "ABORT", "action": "LAND", "success": True},
        {"trigger": "ABORT", "action": "HOLD", "success": False},
    ]
    (run_dir / "safety_actions.jsonl").write_text(
        "\n".join(json.dumps(a) for a in actions),
    )

    result = build_safety_summary(run_dir)
    assert result["all_successful"] is False


def test_build_run_summary(run_dir: Path):
    result = build_run_summary(run_dir)
    assert result["run_id"] == run_dir.name
    assert result["metadata"]["mission_id"] == "test_001"
    assert "events.jsonl" in result["artifact_line_counts"]
    assert result["artifact_line_counts"]["events.jsonl"] == 2


def test_build_run_summary_with_scenario(run_dir: Path):
    scenario = {"scenario_id": "observation_confirmed", "name": "Confirmed Observation"}
    (run_dir / "scenario_metadata.json").write_text(json.dumps(scenario))

    result = build_run_summary(run_dir)
    assert result["scenario"]["scenario_id"] == "observation_confirmed"
