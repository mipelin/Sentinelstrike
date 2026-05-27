"""Metrics structures and automated analysis for simulation tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SimMetrics:
    """Structured metrics from a single simulation test run."""

    test_config: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    detections: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)
    flight: dict = field(default_factory=dict)
    assessment: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "SimMetrics":
        """Load metrics from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Filter to known fields only
        known = {k: data.get(k, field(default_factory=dict).default_factory()) for k in cls.__dataclass_fields__}
        return cls(**known)

    def to_json(self, path: str | Path) -> None:
        """Save metrics to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    def analyze(self) -> list[str]:
        """Return human-readable list of detected issues."""
        issues: list[str] = []
        a = self.assessment
        if a.get("controller_inert"):
            issues.append("Controller inert: drone barely moved (<2m, <0.1 m/s avg)")
        if a.get("target_escaping"):
            issues.append("Target escaping: distance increased while drone stalled")
        if a.get("tracking_unstable"):
            issues.append("Tracking unstable: ID switches or excessive reacquisitions")
        if a.get("world_projection_poor"):
            issues.append("World projection poor: <50% valid frames")
        if a.get("offboard_poor"):
            issues.append("Offboard poor: <50% active during follow")
        return issues

    @property
    def overall(self) -> str:
        """Return overall assessment grade: PASS / WARN / FAIL."""
        return self.assessment.get("overall", "UNKNOWN")
