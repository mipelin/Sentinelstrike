"""Report generation for simulation test suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SimReporter:
    """Collects multiple test runs and generates comparative reports."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs: list[dict] = []

    def add_run(
        self,
        metrics_path: str | Path,
        label: str | None = None,
    ) -> None:
        """Add a metrics JSON file to the report."""
        with open(metrics_path) as f:
            data = json.load(f)
        data["_label"] = label or f"run_{len(self.runs) + 1}"
        data["_source"] = str(metrics_path)
        self.runs.append(data)

    def generate_markdown(self) -> str:
        """Generate a Markdown comparison table."""
        lines: list[str] = [
            "# Auto-Sim Test Report",
            "",
            f"**Total runs:** {len(self.runs)}",
            "",
            "## Summary",
            "",
            "| # | Label | Profile | Spawn | Duration | Frames | Movement | Dist Start | Dist End | Locked% | Assessment |",
            "|---|-------|---------|-------|----------|--------|----------|------------|----------|---------|------------|",
        ]

        for i, run in enumerate(self.runs):
            cfg = run.get("test_config", {})
            flight = run.get("flight", {})
            assess = run.get("assessment", {})
            timing = run.get("timing", {})
            lines.append(
                f"| {i + 1} "
                f"| {run.get('_label', '-')} "
                f"| {cfg.get('standoff_profile', '-')} "
                f"| {cfg.get('spawn_pose', '-')} "
                f"| {timing.get('duration_s', 0):.1f}s "
                f"| {timing.get('frames_processed', 0)} "
                f"| {flight.get('total_xy_movement_m', 0):.1f}m "
                f"| {flight.get('distance_to_target_start_m', '-')} "
                f"| {flight.get('distance_to_target_end_m', '-')} "
                f"| {flight.get('target_locked_pct', 0):.0f}% "
                f"| {assess.get('overall', '-')} |"
            )

        lines.extend(["", "## Issues by Run", ""])
        any_issues = False
        for i, run in enumerate(self.runs):
            issues = run.get("assessment", {}).get("issues", [])
            if issues:
                any_issues = True
                lines.append(f"### Run {i + 1} — {run.get('_label', '')}")
                for issue in issues:
                    lines.append(f"- {issue}")
                lines.append("")
        if not any_issues:
            lines.append("No issues detected across all runs.")
            lines.append("")

        return "\n".join(lines)

    def generate_json(self) -> dict[str, Any]:
        """Generate a JSON report structure."""
        return {
            "runs": self.runs,
            "summary": {
                "total_runs": len(self.runs),
                "pass_count": sum(
                    1 for r in self.runs if r.get("assessment", {}).get("overall") == "PASS"
                ),
                "warn_count": sum(
                    1 for r in self.runs if r.get("assessment", {}).get("overall") == "WARN"
                ),
                "fail_count": sum(
                    1 for r in self.runs if r.get("assessment", {}).get("overall") == "FAIL"
                ),
            },
        }

    def write(self) -> None:
        """Write report.md and report.json to output_dir."""
        md_path = self.output_dir / "report.md"
        md_path.write_text(self.generate_markdown())
        json_path = self.output_dir / "report.json"
        with open(json_path, "w") as f:
            json.dump(self.generate_json(), f, indent=2, default=str)
        print(f"[REPORT] Written to {self.output_dir}")
