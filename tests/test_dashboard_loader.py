"""Tests for dashboard artifact loader."""

import json

from sentinel.dashboard.loader import (
    list_run_dirs,
    load_run_artifacts,
    read_json,
    read_jsonl,
)


def test_read_json_valid(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('{"key": "value"}', encoding="utf-8")
    result = read_json(p)
    assert result == {"key": "value"}


def test_read_json_missing(tmp_path):
    result = read_json(tmp_path / "missing.json")
    assert result is None


def test_read_json_invalid(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    assert read_json(p) is None


def test_read_jsonl_valid(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    result = read_jsonl(p)
    assert len(result) == 2
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}


def test_read_jsonl_missing(tmp_path):
    result = read_jsonl(tmp_path / "missing.jsonl")
    assert result == []


def test_read_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\nbad line\n{"b":2}\n', encoding="utf-8")
    result = read_jsonl(p)
    assert len(result) == 2


def test_load_run_artifacts_no_fail_on_empty(tmp_path):
    artifacts = load_run_artifacts(tmp_path)
    assert artifacts["metadata"] is None
    assert artifacts["detections"] == []
    assert artifacts["tracks"] == []
    assert artifacts["geo_observations"] == []
    assert artifacts["report_md"] is None


def test_load_run_artifacts_with_files(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"mission_id": "m1"}), encoding="utf-8")
    (tmp_path / "detections.jsonl").write_text('{"frame_id":0}\n', encoding="utf-8")
    (tmp_path / "report.md").write_text("# Report", encoding="utf-8")

    artifacts = load_run_artifacts(tmp_path)
    assert artifacts["metadata"]["mission_id"] == "m1"
    assert len(artifacts["detections"]) == 1
    assert artifacts["report_md"] == "# Report"


def test_list_run_dirs(tmp_path):
    run1 = tmp_path / "20260511T001000Z_demo_001"
    run2 = tmp_path / "20260511T002000Z_demo_002"
    run1.mkdir()
    run2.mkdir()
    (run1 / "metadata.json").write_text(json.dumps({"mission_id": "m1"}), encoding="utf-8")
    (run2 / "metadata.json").write_text(json.dumps({"mission_id": "m2"}), encoding="utf-8")
    (run2 / "report.md").write_text("# Report", encoding="utf-8")

    runs = list_run_dirs(tmp_path)
    assert len(runs) == 2
    assert runs[0].run_id == "20260511T002000Z_demo_002"
    assert runs[0].mission_id == "m2"
    assert runs[0].has_report is True
    assert runs[1].has_report is False


def test_list_run_dirs_empty(tmp_path):
    assert list_run_dirs(tmp_path) == []


def test_list_run_dirs_no_dir(tmp_path):
    assert list_run_dirs(tmp_path / "nonexistent") == []
