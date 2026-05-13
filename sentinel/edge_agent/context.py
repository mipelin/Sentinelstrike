"""Edge agent runtime context — holds all subsystem references."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.types import MissionPlan, MissionRequest, VehicleState
from sentinel.config.schema import AppConfig


class EdgeAgentContext:
    """Mutable container for all subsystem references during a mission run."""

    def __init__(
        self,
        config: AppConfig,
        mission_request: MissionRequest,
        event_bus: EventBus,
        mode: str,
    ) -> None:
        self.config = config
        self.mission_request = mission_request
        self.mission_plan: MissionPlan | None = None
        self.run_dir: Path | None = None
        self.event_bus = event_bus
        self.recorder: object | None = None
        self.mavlink_bridge: object | None = None
        self.perception_runner: object | None = None
        self.tracking_runner: object | None = None
        self.geolocalization_runner: object | None = None
        self.tak_bridge: object | None = None
        self.vehicle_state: VehicleState | None = None
        self.mode = mode
        self.started_at_utc: str | None = None
        self.ended_at_utc: str | None = None

    def require_run_dir(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("Run directory not initialized — recorder must be started first")
        return self.run_dir

    def set_vehicle_state(self, state: VehicleState) -> None:
        self.vehicle_state = state

    def close_all(self) -> None:
        """Close all subsystems. Idempotent — safe to call multiple times."""
        closables = [
            ("tak_bridge", self.tak_bridge),
            ("geolocalization_runner", self.geolocalization_runner),
            ("tracking_runner", self.tracking_runner),
            ("recorder", self.recorder),
            ("mavlink_bridge", self.mavlink_bridge),
        ]
        for name, obj in closables:
            if obj is not None:
                try:
                    obj.close()  # type: ignore[union-attr]
                    logger.debug("Closed {}", name)
                except Exception as exc:
                    logger.warning("Error closing {}: {}", name, exc)
