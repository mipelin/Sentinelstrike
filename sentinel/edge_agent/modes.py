"""Edge agent operating modes."""

from __future__ import annotations

from enum import StrEnum


class EdgeAgentMode(StrEnum):
    MOCK = "mock"
    PX4_SITL = "px4_sitl"


VALID_MODES = {m.value for m in EdgeAgentMode}


def validate_mode(mode: str) -> EdgeAgentMode:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid edge agent mode '{mode}'. Valid: {sorted(VALID_MODES)}")
    return EdgeAgentMode(mode)
