"""Tests for smoke checks."""

from pathlib import Path

from sentinel.quality.smoke import run_basic_smoke_checks


def test_smoke_checks_on_current_project():
    result = run_basic_smoke_checks(Path("."))
    assert result["valid"] is True
    assert len(result["checks"]) > 0


def test_smoke_checks_missing_root(tmp_path):
    result = run_basic_smoke_checks(tmp_path)
    assert result["valid"] is False
    assert len(result["errors"]) > 0
