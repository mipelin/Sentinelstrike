"""Safety policy — maps triggers to allowed actions with configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .actions import SafetyAction, SafetyTrigger

_ABORT_ACTIONS = [SafetyAction.HOLD, SafetyAction.LAND]
_RETURN_TO_SAFE_ACTIONS = [SafetyAction.RETURN_HOME]


class SafetyPolicyConfig(BaseModel):
    abort_sequence: list[SafetyAction] = Field(
        default=[SafetyAction.HOLD, SafetyAction.LAND],
        description="Ordered sequence of actions for OPERATOR_ABORT trigger.",
    )
    return_to_safe_action: SafetyAction = Field(
        default=SafetyAction.RETURN_HOME,
        description="Action for OPERATOR_RETURN_TO_SAFE trigger.",
    )
    link_loss_action: SafetyAction = Field(
        default=SafetyAction.RETURN_HOME,
        description="Action for LINK_LOSS trigger.",
    )
    low_battery_threshold_pct: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Battery percentage below which LOW_BATTERY triggers.",
    )
    low_battery_action: SafetyAction = Field(
        default=SafetyAction.RETURN_HOME,
        description="Action for LOW_BATTERY trigger.",
    )
    telemetry_stale_threshold: int = Field(
        default=5,
        ge=1,
        description="Number of consecutive stale telemetry reads before triggering.",
    )
    telemetry_stale_action: SafetyAction = Field(
        default=SafetyAction.HOLD,
        description="Action for TELEMETRY_STALE trigger.",
    )
    allow_disarm: bool = Field(
        default=False,
        description="Whether DISARM is allowed as a safety action.",
    )


class SafetyPolicyV1:
    """Maps safety triggers to actions and validates them."""

    def __init__(self, config: SafetyPolicyConfig | None = None) -> None:
        self._config = config or SafetyPolicyConfig()

    @property
    def config(self) -> SafetyPolicyConfig:
        return self._config

    def resolve_action(self, trigger: SafetyTrigger) -> SafetyAction | list[SafetyAction]:
        """Return the action(s) for a given trigger."""
        match trigger:
            case SafetyTrigger.OPERATOR_ABORT:
                return list(self._config.abort_sequence)
            case SafetyTrigger.OPERATOR_RETURN_TO_SAFE:
                return self._config.return_to_safe_action
            case SafetyTrigger.LINK_LOSS:
                return self._config.link_loss_action
            case SafetyTrigger.LOW_BATTERY:
                return self._config.low_battery_action
            case SafetyTrigger.TELEMETRY_STALE:
                return self._config.telemetry_stale_action
            case SafetyTrigger.TELEMETRY_INVALID:
                return SafetyAction.HOLD
            case SafetyTrigger.MANUAL:
                return SafetyAction.HOLD
            case _:
                return SafetyAction.HOLD

    def is_action_allowed(self, action: SafetyAction) -> bool:
        """Check whether an action is permitted under current policy."""
        if action == SafetyAction.DISARM:
            return self._config.allow_disarm
        if action == SafetyAction.NO_ACTION:
            return True
        return action in (
            SafetyAction.HOLD,
            SafetyAction.RETURN_HOME,
            SafetyAction.LAND,
        )

    def check_battery(self, battery_pct: float | None) -> bool:
        """Return True if battery is below threshold (triggers LOW_BATTERY)."""
        if battery_pct is None:
            return False
        return battery_pct < self._config.low_battery_threshold_pct

    def check_telemetry_stale(self, stale_count: int) -> bool:
        """Return True if stale count exceeds threshold (triggers TELEMETRY_STALE)."""
        return stale_count >= self._config.telemetry_stale_threshold
