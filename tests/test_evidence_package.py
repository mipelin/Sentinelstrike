"""Tests for evidence package builder and exporter."""

import json
import zipfile
from pathlib import Path

from sentinel.evidence.exporter import export_zip
from sentinel.evidence.package_builder import PACKAGE_DIR_NAME, build_evidence_package


def _make_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "20260511T120000Z_test_run"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"mission_id": "test_001"}))
    (d / "events.jsonl").write_text(
        '{"timestamp_utc":"2026-05-11T12:00:01Z","event_type":"start"}\n',
    )
    (d / "detections.jsonl").write_text('{"frame_id":1}\n')
    (d / "tracks.jsonl").write_text('{"track_id":"trk_001"}\n')
    (d / "report.md").write_text("# Report\n")
    (d / "perception_metrics.json").write_text(json.dumps({"frame_count": 10}))
    (d / "tracker_metrics.json").write_text(json.dumps({"total_tracks": 5}))
    return d


def test_build_evidence_package_creates_directory(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    pkg_dir = build_evidence_package(run_dir)

    assert pkg_dir.exists()
    assert pkg_dir.name == PACKAGE_DIR_NAME


def test_build_evidence_package_has_all_files(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    pkg_dir = build_evidence_package(run_dir)

    expected = {
        "summary.json",
        "metrics_summary.json",
        "operator_summary.json",
        "safety_summary.json",
        "timeline.json",
        "replay_manifest.json",
        "artifacts_index.json",
        "checksums.json",
    }
    actual = {p.name for p in pkg_dir.iterdir() if p.is_file()}
    assert expected <= actual


def test_summary_json_valid(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    build_evidence_package(run_dir)

    summary = json.loads((run_dir / PACKAGE_DIR_NAME / "summary.json").read_text())
    assert summary["run_id"] == run_dir.name
    assert summary["metadata"]["mission_id"] == "test_001"
    assert "artifact_line_counts" in summary


def test_metrics_summary_json_valid(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    build_evidence_package(run_dir)

    metrics = json.loads((run_dir / PACKAGE_DIR_NAME / "metrics_summary.json").read_text())
    assert "perception_metrics" in metrics
    assert metrics["perception_metrics"]["frame_count"] == 10


def test_replay_manifest_json_valid(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    build_evidence_package(run_dir)

    manifest = json.loads((run_dir / PACKAGE_DIR_NAME / "replay_manifest.json").read_text())
    assert manifest["run_id"] == run_dir.name
    assert manifest["artifact_count"] > 0
    assert len(manifest["artifacts"]) > 0


def test_checksums_json_valid(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    build_evidence_package(run_dir)

    checksums = json.loads((run_dir / PACKAGE_DIR_NAME / "checksums.json").read_text())
    assert len(checksums) > 0
    for name, sha in checksums.items():
        assert len(sha) == 64
        assert (run_dir / PACKAGE_DIR_NAME / name).exists()


def test_timeline_json_is_sorted(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    build_evidence_package(run_dir)

    timeline = json.loads((run_dir / PACKAGE_DIR_NAME / "timeline.json").read_text())
    assert isinstance(timeline, list)
    timestamps = [e.get("timestamp_utc", "") for e in timeline]
    assert timestamps == sorted(timestamps)


def test_export_zip_creates_archive(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    build_evidence_package(run_dir)

    zip_path = export_zip(run_dir)
    assert zip_path is not None
    assert zip_path.exists()
    assert zip_path.name == "mission_package.zip"

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert any("summary.json" in n for n in names)
        assert any("replay_manifest.json" in n for n in names)


def test_export_zip_returns_none_without_package(tmp_path: Path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    assert export_zip(run_dir) is None


def test_evidence_package_idempotent(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    pkg1 = build_evidence_package(run_dir)
    pkg2 = build_evidence_package(run_dir)

    assert pkg1 == pkg2
    files = list(pkg1.iterdir())
    assert len(files) == 8
