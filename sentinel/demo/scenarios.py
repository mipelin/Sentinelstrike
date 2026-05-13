"""Demo scenario definitions."""

from __future__ import annotations

from pydantic import BaseModel


class DemoScenario(BaseModel):
    scenario_id: str
    name: str
    description: str
    mode: str = "mock"
    backend: str = "mock"
    tak_mode: str = "dry_run"
    max_frames: int = 30
    target_fps: float = 10.0
    operator_mode: str = "confirm_above_threshold"
    safety_trigger: str | None = None
    simulated_battery_pct: float | None = None
    simulated_link_ok: bool = True
    requires_px4: bool = False
