"""Tests for the stability test tool."""

from __future__ import annotations

import json
from pathlib import Path


def test_dry_run_produces_valid_report(tmp_path, monkeypatch):
    """Dry run with 0.01 min duration should produce a valid report."""
    monkeypatch.chdir(tmp_path)
    from apps.tools.run_stability_test import run_stability_test

    report = run_stability_test(
        duration_min=0.01,
        backend="mock",
        target_fps=5.0,
        save_frames=False,
        report_every_s=999,
    )
    assert "frame_count" in report
    assert "actual_fps" in report
    assert "unique_tracks" in report
    assert "id_churn_rate" in report
    assert "average_tick_ms" in report
    assert report["frame_count"] >= 0
    assert report["duration_min"] == 0.01


def test_report_schema(tmp_path, monkeypatch):
    """Report JSON should have all required fields."""
    monkeypatch.chdir(tmp_path)
    from apps.tools.run_stability_test import run_stability_test

    report = run_stability_test(
        duration_min=0.01,
        backend="mock",
        target_fps=5.0,
        save_frames=False,
    )
    required_fields = [
        "duration_min", "backend", "target_fps", "actual_fps",
        "frame_count", "dropped_frames", "unique_tracks",
        "id_churn_rate", "average_tick_ms", "min_tick_ms", "max_tick_ms",
        "wall_clock_s", "errors",
    ]
    for field in required_fields:
        assert field in report, f"Missing field: {field}"


def test_no_frames_when_disabled(tmp_path, monkeypatch):
    """With save_frames=False, no frame images should be written."""
    monkeypatch.chdir(tmp_path)
    from apps.tools.run_stability_test import run_stability_test

    report = run_stability_test(
        duration_min=0.01,
        backend="mock",
        target_fps=5.0,
        save_frames=False,
    )
    # Find the run dir
    runs_dir = tmp_path / "runs"
    if runs_dir.exists():
        frames_dir = runs_dir / "stability_test" / "frames"
        if not frames_dir.exists():
            # Look for the actual stability_* dir
            for d in runs_dir.iterdir():
                fd = d / "frames"
                if fd.exists():
                    jpgs = list(fd.glob("*.jpg"))
                    assert len(jpgs) == 0, "Frames were saved despite save_frames=False"
