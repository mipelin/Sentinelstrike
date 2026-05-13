"""Mission Evidence Package builder — orchestrates all evidence generation."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from .manifest import build_artifacts_index, build_consolidated_timeline, build_replay_manifest
from .summaries import build_metrics_summary, build_operator_summary, build_run_summary, build_safety_summary

PACKAGE_DIR_NAME = "mission_package"


def build_evidence_package(run_dir: Path) -> Path:
    """Build the complete evidence package for a run directory.

    Creates a mission_package/ subdirectory inside run_dir with:
      - summary.json
      - metrics_summary.json
      - operator_summary.json
      - safety_summary.json
      - timeline.json
      - replay_manifest.json
      - checksums.json
      - artifacts_index.json
    """
    run_dir = Path(run_dir)
    pkg_dir = run_dir / PACKAGE_DIR_NAME
    pkg_dir.mkdir(exist_ok=True)

    logger.info("Building evidence package: {}", pkg_dir)

    # Summaries
    _write(pkg_dir / "summary.json", build_run_summary(run_dir))
    _write(pkg_dir / "metrics_summary.json", build_metrics_summary(run_dir))
    _write(pkg_dir / "operator_summary.json", build_operator_summary(run_dir))
    _write(pkg_dir / "safety_summary.json", build_safety_summary(run_dir))

    # Timeline
    _write(pkg_dir / "timeline.json", build_consolidated_timeline(run_dir))

    # Manifest and index
    _write(pkg_dir / "replay_manifest.json", build_replay_manifest(run_dir))
    _write(pkg_dir / "artifacts_index.json", build_artifacts_index(run_dir))

    # Checksums of package files
    from .checksums import compute_checksums
    _write(pkg_dir / "checksums.json", compute_checksums(pkg_dir))

    logger.info("Evidence package complete: {} files", len(list(pkg_dir.iterdir())))
    return pkg_dir


def _write(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
