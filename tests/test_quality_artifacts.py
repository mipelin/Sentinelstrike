"""Tests for artifact validation."""

import json

from sentinel.quality.artifacts import (
    EXPECTED_PIPELINE_ARTIFACTS,
    find_latest_run_dir,
    validate_pipeline_run_dir,
)


def _make_full_run_dir(tmp_path):
    """Create a minimal valid run directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in EXPECTED_PIPELINE_ARTIFACTS:
        p = run_dir / name
        if name.endswith(".jsonl"):
            p.write_text('{"test":1}\n', encoding="utf-8")
        elif name.endswith(".json"):
            if name == "pipeline_summary.json":
                data = {"mission_id": "m1", "planner": {}, "artifacts": {}}
            else:
                data = {"ok": True}
            p.write_text(json.dumps(data), encoding="utf-8")
        elif name.endswith(".md"):
            p.write_text("# Report\nContent here.", encoding="utf-8")
    return run_dir


def test_complete_run_dir_valid(tmp_path):
    run_dir = _make_full_run_dir(tmp_path)
    result = validate_pipeline_run_dir(run_dir)
    assert result["valid"] is True
    assert result["missing"] == []


def test_detects_missing_artifact(tmp_path):
    run_dir = _make_full_run_dir(tmp_path)
    (run_dir / "tracks.jsonl").unlink()
    result = validate_pipeline_run_dir(run_dir)
    assert result["valid"] is False
    assert "tracks.jsonl" in result["missing"]


def test_detects_invalid_json(tmp_path):
    run_dir = _make_full_run_dir(tmp_path)
    (run_dir / "metadata.json").write_text("not json", encoding="utf-8")
    result = validate_pipeline_run_dir(run_dir)
    assert result["valid"] is False
    assert any("metadata.json" in e for e in result["json_errors"])


def test_detects_invalid_jsonl(tmp_path):
    run_dir = _make_full_run_dir(tmp_path)
    (run_dir / "detections.jsonl").write_text("not json\n", encoding="utf-8")
    result = validate_pipeline_run_dir(run_dir)
    assert result["valid"] is False


def test_find_latest_run_dir(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()
    (base / "20260101T000000Z_m1").mkdir()
    (base / "20260102T000000Z_m2").mkdir()
    (base / ".gitkeep").touch()
    latest = find_latest_run_dir(base)
    assert latest is not None
    assert latest.name == "20260102T000000Z_m2"


def test_find_latest_run_dir_empty(tmp_path):
    base = tmp_path / "empty"
    base.mkdir()
    assert find_latest_run_dir(base) is None
