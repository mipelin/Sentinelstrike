"""Demo scenario runner — executes scenarios end-to-end."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.types import GeoPoint, VehicleState
from sentinel.config.loader import load_config
from sentinel.config.schema import AppConfig
from sentinel.mavlink_bridge.bridge import MavlinkBridge
from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
from sentinel.operator.gate import OperatorDecisionGate
from sentinel.operator.policy import OperatorPolicy
from sentinel.operator.simulator import OperatorSimulator
from sentinel.realtime.loop import RealTimeMissionLoop
from sentinel.recorder.recorder import MissionRecorder
from sentinel.safety.executor import SafetyActionExecutor
from sentinel.safety.policy import SafetyPolicyV1

from .scenarios import DemoScenario


def run_scenario(
    scenario: DemoScenario,
    config_path: str = "configs/sim.yaml",
) -> dict:
    """Run a demo scenario and return summary dict."""
    cfg = load_config(config_path)
    event_bus = EventBus()

    mission_id = f"demo_{scenario.scenario_id}"
    recorder = MissionRecorder(cfg.recorder.output_dir, mission_id)
    run_dir = recorder.start_run()
    event_bus.subscribe("*", recorder.record_event)

    logger.info("Running demo scenario: {} ({})", scenario.scenario_id, scenario.name)

    # Perception
    perception_backend = _make_perception(cfg, scenario)

    # Tracking
    from sentinel.tracker.runner import TrackingRunner
    tracker_runner = TrackingRunner(
        config=cfg.tracker, mission_id=mission_id, event_bus=event_bus, run_dir=run_dir,
    )

    # Geolocalization
    from sentinel.geolocalizer.runner import GeolocalizationRunner
    geo_runner = GeolocalizationRunner(
        config=cfg.geolocalizer, mission_id=mission_id, event_bus=event_bus, run_dir=run_dir,
    )

    # TAK
    from sentinel.tak_bridge.bridge import TakBridge
    tak_bridge = TakBridge(
        config=cfg.tak.model_copy(update={"mode": scenario.tak_mode}),
        mission_id=mission_id, event_bus=event_bus, run_dir=run_dir,
    )

    # MAVLink bridge (mock for all non-PX4 scenarios)
    mock_backend = MockMavlinkBackend(vehicle_id=cfg.system.vehicle_id)
    mavlink_bridge = MavlinkBridge(
        mission_id=mission_id, vehicle_id=cfg.system.vehicle_id,
        backend=mock_backend, event_bus=event_bus,
    )

    # Operator gate
    op_cfg = cfg.operator
    op_policy = OperatorPolicy(
        min_track_confidence_for_confirmation=op_cfg.min_track_confidence_for_confirmation,
        min_observation_confidence_for_orbit=op_cfg.min_observation_confidence_for_orbit,
        require_human_for_orbit=op_cfg.require_human_for_orbit,
        require_human_for_abort=op_cfg.require_human_for_abort,
        auto_reject_below_confidence=op_cfg.auto_reject_below_confidence,
    )
    operator_gate = OperatorDecisionGate(
        mission_id=mission_id, policy=op_policy, event_bus=event_bus,
        run_dir=run_dir, operator_id=op_cfg.operator_id,
    )
    operator_simulator = OperatorSimulator(
        mode=scenario.operator_mode, operator_id=op_cfg.operator_id, policy=op_policy,
    )

    # Safety executor
    safety_policy = SafetyPolicyV1()
    safety_executor = SafetyActionExecutor(
        mission_id=mission_id, policy=safety_policy,
        mavlink_bridge=mavlink_bridge, event_bus=event_bus, run_dir=run_dir,
    )

    # Inject scenario-specific safety triggers before the loop
    _inject_safety_trigger(scenario, safety_executor, operator_gate, operator_simulator)

    # Realtime loop
    loop = RealTimeMissionLoop(
        config=cfg,
        mission_id=mission_id,
        event_bus=event_bus,
        run_dir=run_dir,
        perception_backend=perception_backend,
        tracker_runner=tracker_runner,
        geolocalization_runner=geo_runner,
        tak_bridge=tak_bridge,
        mavlink_bridge=mavlink_bridge if scenario.mode == "px4_sitl" else None,
        max_frames=scenario.max_frames,
        target_fps=scenario.target_fps,
        video_source=cfg.perception.source,
        operator_gate=operator_gate,
        operator_simulator=operator_simulator,
        safety_executor=safety_executor,
    )

    try:
        metrics = loop.run()
    finally:
        recorder.close()

    # Write scenario metadata
    summary = {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "executed_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    (run_dir / "scenario_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    # Build evidence package
    from sentinel.evidence.package_builder import build_evidence_package
    build_evidence_package(run_dir)

    logger.info(
        "Scenario {} complete: {} frames, {} detections, {} observations",
        scenario.scenario_id, metrics.get("frame_count", 0),
        metrics.get("detection_count", 0), metrics.get("geo_observation_count", 0),
    )

    return summary


def _make_perception(cfg: AppConfig, scenario: DemoScenario):
    from sentinel.perception.mock_backend import MockPerceptionBackend

    perc_cfg = cfg.perception.model_copy(
        update={"backend": scenario.backend, "max_frames": scenario.max_frames},
    )
    if scenario.backend == "mock":
        return MockPerceptionBackend(
            classes=perc_cfg.classes,
            confidence_threshold=perc_cfg.confidence_threshold,
        )
    from sentinel.perception.yolo_backend import YoloPerceptionBackend
    return YoloPerceptionBackend(
        model_path=perc_cfg.model_path, classes=perc_cfg.classes,
        confidence_threshold=perc_cfg.confidence_threshold,
        device=perc_cfg.device, image_size=perc_cfg.image_size,
    )


def _inject_safety_trigger(
    scenario: DemoScenario,
    safety_executor: SafetyActionExecutor,
    operator_gate: OperatorDecisionGate,
    operator_simulator: OperatorSimulator,
) -> None:
    """Pre-inject scenario-specific safety conditions for the loop to pick up."""
    if scenario.safety_trigger == "abort":
        # Generate an ABORT decision so the gate sets abort_requested=True
        from sentinel.operator.decisions import OperatorAction, make_decision_record
        rec = make_decision_record(
            mission_id=safety_executor._mission_id,
            operator_id="demo_scenario",
            action=OperatorAction.ABORT_MISSION,
            reason="demo scenario: operator abort",
        )
        operator_gate.record_decision(rec)

    if scenario.safety_trigger == "low_battery":
        # Fire low battery directly
        safety_executor.evaluate_system_conditions(
            vehicle_state=VehicleState(
                vehicle_id="uav_001",
                timestamp_utc="2026-01-01T00:00:00Z",
                position=GeoPoint(lat=38.0, lon=-8.0, alt_m=50.0),
                battery_pct=scenario.simulated_battery_pct or 10.0,
            ),
        )

    if scenario.safety_trigger == "link_loss":
        safety_executor.evaluate_system_conditions(
            vehicle_state=VehicleState(
                vehicle_id="uav_001",
                timestamp_utc="2026-01-01T00:00:00Z",
                position=GeoPoint(lat=38.0, lon=-8.0, alt_m=50.0),
                battery_pct=80.0,
            ),
            link_lost=True,
        )
