"""Tests for evidence manifest and artifacts index."""

import json
from pathlib import Path

from sentinel.evidence.manifest import (
    build_artifacts_index,
    build_consolidated_timeline,
    build_replay_manifest,
)


def _make_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "20260511T120000Z_test_run"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"mission_id": "test_001"}))
    (d / "events.jsonl").write_text(
        '{"timestamp_utc":"2026-05-11T12:00:01Z","event_type":"start"}\n'
        '{"timestamp_utc":"2026-05-11T12:00:02Z","event_type":"end"}\n',
    )
    (d / "detections.jsonl").write_text('{"frame_id":1}\n')
    (d / "tracks.jsonl").write_text('{"track_id":"trk_001"}\n')
    (d / "geo_observations.jsonl").write_text('{"observation_id":"obs_001","timestamp_utc":"2026-05-11T12:00:03Z"}\n')
    (d / "operator_decisions.jsonl").write_text('{"action":"CONFIRM","timestamp_utc":"2026-05-11T12:00:02Z"}\n')
    (d / "safety_actions.jsonl").write_text('{"trigger":"LOW_BATTERY","timestamp_utc":"2026-05-11T12:00:04Z"}\n')
    (d / "report.md").write_text("# Report\n")
    (d / "pipeline_summary.json").write_text(json.dumps({"mission_id": "test_001"}))
    return d


def test_replay_manifest_has_all_fields(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    manifest = build_replay_manifest(run_dir)

    assert manifest["replay_version"] == "1.0"
    assert manifest["run_id"] == run_dir.name
    assert manifest["mission_id"] == "test_001"
    assert manifest["artifact_count"] > 0
    assert manifest["total_size_bytes"] > 0
    assert len(manifest["artifacts"]) > 0

    for a in manifest["artifacts"]:
        assert "filename" in a
        assert "size_bytes" in a
        assert "sha256" in a
        assert len(a["sha256"]) == 64


def test_replay_manifest_checksums_are_consistent(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    manifest = build_replay_manifest(run_dir)

    for artifact in manifest["artifacts"]:
        if artifact["filename"].endswith(".json"):
            content = (run_dir / artifact["filename"]).read_bytes()
            import hashlib
            expected = hashlib.sha256(content).hexdigest()
            assert artifact["sha256"] == expected


def test_artifacts_index_categories(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    index = build_artifacts_index(run_dir)

    assert "events" in index
    assert any(a["filename"] == "events.jsonl" for a in index["events"])
    assert "perception" in index
    assert any(a["filename"] == "detections.jsonl" for a in index["perception"])
    assert "tracking" in index
    assert "geolocalization" in index
    assert "operator" in index
    assert "safety" in index
    assert "report" in index


def test_artifacts_index_no_empty_categories(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    index = build_artifacts_index(run_dir)

    for cat, items in index.items():
        assert len(items) > 0, f"Category {cat} should not be empty"


def test_consolidated_timeline_sorted(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    timeline = build_consolidated_timeline(run_dir)

    timestamps = [e.get("timestamp_utc", "") for e in timeline]
    assert timestamps == sorted(timestamps)


def test_consolidated_timeline_has_source_types(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    timeline = build_consolidated_timeline(run_dir)

    source_types = {e["_source_type"] for e in timeline}
    assert "system_event" in source_types
    assert "operator_decision" in source_types
    assert "safety_action" in source_types
    assert "geo_observation" in source_types
