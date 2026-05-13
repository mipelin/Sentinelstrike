"""Basic smoke checks for project integrity."""

from __future__ import annotations

from pathlib import Path

REQUIRED_FILES = [
    "configs/sim.yaml",
    "missions/demo_search_area.json",
    "README.md",
    "pyproject.toml",
]

REQUIRED_IMPORTS = [
    ("sentinel.common.types", "GeoPoint"),
    ("sentinel.mission_planner.planner", "MissionPlanner"),
    ("sentinel.perception.detector", "PerceptionRunner"),
    ("sentinel.tracker.simple_tracker", "SimpleIoUTracker"),
    ("sentinel.geolocalizer.flat_ground", "FlatGroundGeolocalizer"),
    ("sentinel.tak_bridge.bridge", "TakBridge"),
    ("sentinel.pipeline.integrated_pipeline", "IntegratedMissionPipeline"),
]


def run_basic_smoke_checks(project_root: Path) -> dict:
    checks: list[dict] = []
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        p = project_root / rel
        ok = p.exists()
        checks.append({"check": f"file_exists:{rel}", "passed": ok})
        if not ok:
            errors.append(f"Missing file: {rel}")

    for module_name, attr in REQUIRED_IMPORTS:
        try:
            mod = __import__(module_name, fromlist=[attr])
            getattr(mod, attr)
            checks.append({"check": f"import:{module_name}.{attr}", "passed": True})
        except Exception as exc:
            checks.append({"check": f"import:{module_name}.{attr}", "passed": False})
            errors.append(f"Import failed: {module_name}.{attr}: {exc}")

    return {"valid": len(errors) == 0, "checks": checks, "errors": errors}
