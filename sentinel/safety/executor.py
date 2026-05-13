"""Safety Action Executor — evaluates conditions and executes safety commands via MavlinkBridge."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.types import VehicleState

from .actions import SafetyAction, SafetyActionRecord, SafetyTrigger, make_safety_action_record
from .policy import SafetyPolicyV1


class SafetyActionExecutor:
    """Evaluates operator decisions and system conditions, then executes safety actions.

    One-shot: each trigger fires at most once per mission. Call reset() to re-arm.
    """

    def __init__(
        self,
        mission_id: str,
        policy: SafetyPolicyV1,
        mavlink_bridge: object,
        event_bus: EventBus | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self._mission_id = mission_id
        self._policy = policy
        self._bridge = mavlink_bridge
        self._event_bus = event_bus
        self._run_dir = run_dir

        self._records: list[SafetyActionRecord] = []
        self._fired_triggers: set[SafetyTrigger] = set()
        self._action_log_path = run_dir / "safety_actions.jsonl" if run_dir else None

    @property
    def policy(self) -> SafetyPolicyV1:
        return self._policy

    @property
    def records(self) -> list[SafetyActionRecord]:
        return list(self._records)

    @property
    def abort_executed(self) -> bool:
        return SafetyTrigger.OPERATOR_ABORT in self._fired_triggers

    @property
    def return_to_safe_executed(self) -> bool:
        return SafetyTrigger.OPERATOR_RETURN_TO_SAFE in self._fired_triggers

    def _execute_bridge_action(self, action: SafetyAction) -> tuple[bool, str]:
        """Dispatch a single safety action to the MAVLink bridge."""
        try:
            if action == SafetyAction.HOLD:
                result = self._bridge.hold()
            elif action == SafetyAction.RETURN_HOME:
                result = self._bridge.return_home()
            elif action == SafetyAction.LAND:
                result = self._bridge.land()
            elif action == SafetyAction.DISARM:
                result = self._bridge.disarm()
            else:
                return False, f"Unsupported safety action: {action}"
            return result.success, result.message
        except Exception as exc:
            return False, str(exc)

    def _execute_trigger(
        self,
        trigger: SafetyTrigger,
        extra_metadata: dict | None = None,
    ) -> list[SafetyActionRecord]:
        """Resolve and execute action(s) for a trigger. Returns records."""
        if trigger in self._fired_triggers:
            return []

        self._fired_triggers.add(trigger)
        resolved = self._policy.resolve_action(trigger)

        if isinstance(resolved, SafetyAction):
            actions = [resolved]
        else:
            actions = resolved

        records: list[SafetyActionRecord] = []
        for action in actions:
            if not self._policy.is_action_allowed(action):
                rec = make_safety_action_record(
                    mission_id=self._mission_id,
                    trigger=trigger,
                    action=action,
                    command_success=False,
                    command_message=f"Action {action} blocked by safety policy",
                    metadata=extra_metadata,
                )
                records.append(rec)
                self._publish_event("safety_action_blocked", trigger, action, rec)
                logger.warning("Safety action {} blocked by policy for trigger {}", action, trigger)
                continue

            success, message = self._execute_bridge_action(action)
            rec = make_safety_action_record(
                mission_id=self._mission_id,
                trigger=trigger,
                action=action,
                command_success=success,
                command_message=message,
                metadata=extra_metadata,
            )
            records.append(rec)
            self._append_record(rec)

            event_type = "safety_action_executed" if success else "safety_action_failed"
            self._publish_event(event_type, trigger, action, rec)
            logger.info("Safety action {} for trigger {} -> success={} msg={}", action, trigger, success, message)

        self._records.extend(records)
        return records

    def _publish_event(
        self, event_type: str, trigger: SafetyTrigger, action: SafetyAction, rec: SafetyActionRecord,
    ) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            make_event(
                mission_id=self._mission_id,
                source="safety_executor",
                event_type=event_type,
                payload={
                    "trigger": str(trigger),
                    "action": str(action),
                    "record_id": rec.record_id,
                    "success": rec.command_success,
                    "message": rec.command_message,
                },
            )
        )

    def _append_record(self, rec: SafetyActionRecord) -> None:
        if self._action_log_path is None:
            return
        with open(self._action_log_path, "a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")

    # --- public trigger methods ---

    def evaluate_operator_decision(self, abort_requested: bool, return_to_safe_requested: bool) -> list[SafetyActionRecord]:
        """Check operator gate flags and fire if set."""
        results: list[SafetyActionRecord] = []
        if abort_requested:
            results.extend(self._execute_trigger(SafetyTrigger.OPERATOR_ABORT))
        if return_to_safe_requested:
            results.extend(self._execute_trigger(SafetyTrigger.OPERATOR_RETURN_TO_SAFE))
        return results

    def evaluate_system_conditions(
        self,
        vehicle_state: VehicleState | None,
        stale_count: int = 0,
        link_lost: bool = False,
    ) -> list[SafetyActionRecord]:
        """Check telemetry, battery, and link conditions."""
        results: list[SafetyActionRecord] = []

        if link_lost:
            results.extend(self._execute_trigger(SafetyTrigger.LINK_LOSS))

        if vehicle_state is not None:
            if self._policy.check_battery(vehicle_state.battery_pct):
                results.extend(
                    self._execute_trigger(
                        SafetyTrigger.LOW_BATTERY,
                        extra_metadata={"battery_pct": vehicle_state.battery_pct},
                    )
                )
        else:
            results.extend(self._execute_trigger(SafetyTrigger.TELEMETRY_INVALID))

        if self._policy.check_telemetry_stale(stale_count):
            results.extend(
                self._execute_trigger(
                    SafetyTrigger.TELEMETRY_STALE,
                    extra_metadata={"stale_count": stale_count},
                )
            )

        return results

    def fire_manual(self, action: SafetyAction) -> SafetyActionRecord:
        """Manually trigger a specific safety action."""
        self._fired_triggers.add(SafetyTrigger.MANUAL)
        success, message = self._execute_bridge_action(action)
        rec = make_safety_action_record(
            mission_id=self._mission_id,
            trigger=SafetyTrigger.MANUAL,
            action=action,
            command_success=success,
            command_message=message,
        )
        self._records.append(rec)
        self._append_record(rec)
        self._publish_event(
            "safety_action_executed" if success else "safety_action_failed",
            SafetyTrigger.MANUAL, action, rec,
        )
        return rec

    def reset(self) -> None:
        """Re-arm all triggers (e.g., for a new mission)."""
        self._fired_triggers.clear()

    def close(self) -> dict:
        """Flush and return safety metrics."""
        from .metrics import compute_safety_metrics

        metrics = compute_safety_metrics(self._records)
        if self._run_dir is not None:
            metrics_path = self._run_dir / "safety_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        self._publish_event("safety_executor_completed", SafetyTrigger.MANUAL, SafetyAction.NO_ACTION, make_safety_action_record(
            mission_id=self._mission_id,
            trigger=SafetyTrigger.MANUAL,
            action=SafetyAction.NO_ACTION,
            command_success=True,
            command_message="safety executor closed",
            metadata=metrics,
        ))
        return metrics
