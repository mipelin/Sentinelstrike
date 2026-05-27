"""Simulation-only autonomous Sentinel demo runner.

Launches PX4 SITL + Gazebo, validates sim health, then runs the
YOLO + ISR tracker + standoff follow demo end to end with structured logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sentinel.testing.auto_pilot import AutoPilot
from sentinel.testing.reporter import SimReporter
from sentinel.testing.stack_launcher import SimStackLauncher


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _python_executable(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _require_simulation_only(args: argparse.Namespace) -> None:
    if not args.simulation_only:
        print("SIMULATION ONLY — DO NOT USE WITH REAL HARDWARE", file=sys.stderr)
        print("Refusing to run without --simulation-only.", file=sys.stderr)
        sys.exit(2)


def _world_topic(launcher: SimStackLauncher, world: str, topic: str | None) -> str:
    if topic:
        return topic
    return launcher.camera_topic(world)


def _output_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = repo_root / "logs" / f"sentinel_sim_auto_{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_test_cmd(
    args: argparse.Namespace,
    python_executable: str,
    topic: str,
    annotated_dir: Path,
) -> list[str]:
    max_frames = max(1, int(math.ceil(args.duration * args.target_fps)))
    cmd = [
        python_executable,
        "-m",
        "apps.tools.run_gazebo_yolo_test",
        "--topic",
        topic,
        "--backend",
        "yolo",
        "--track",
        "isr",
        "--follow-mode",
        "standoff",
        "--standoff-distance",
        str(args.standoff_distance),
        "--standoff-altitude",
        str(args.standoff_altitude),
        "--target-fps",
        str(args.target_fps),
        "--max-frames",
        str(max_frames),
        "--timeout",
        str(args.camera_timeout),
        "--use-workers",
        "--auto-follow-nearest-person",
        "--profile",
        "--save-annotated",
        str(annotated_dir),
        "--mavlink-url",
        args.mavlink_url,
        "--model-path",
        args.model_path,
        "--device",
        args.device,
    ]
    if args.headless:
        cmd.extend(["--no-display"])
    return cmd


async def _auto_takeoff(mavlink_url: str, altitude_m: float = 20.0, timeout_s: float = 60.0) -> bool:
    """Arm and takeoff using MAVSDK."""
    from mavsdk import System
    drone = System()
    await drone.connect(system_address=mavlink_url)
    print("  [Takeoff] Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("  [Takeoff] Connected.")
    print("  [Takeoff] Waiting for global position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break
    print("  [Takeoff] Arming...")
    await drone.action.arm()
    print(f"  [Takeoff] Taking off to {altitude_m}m...")
    await drone.action.set_takeoff_altitude(altitude_m)
    await drone.action.takeoff()
    start = time.monotonic()
    async for pos in drone.telemetry.position():
        if pos.relative_altitude_m >= altitude_m * 0.8:
            print(f"  [Takeoff] Reached {pos.relative_altitude_m:.1f}m")
            return True
        if time.monotonic() - start > timeout_s:
            print("  [Takeoff] Timeout waiting for altitude")
            return False
    return False


def _stream_subprocess(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, timeout_s: float) -> int:
    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write("$ " + shlex.join(cmd) + "\n")
        log_fp.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        start = time.monotonic()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_fp.write(line)
                log_fp.flush()
                if time.monotonic() - start > timeout_s:
                    proc.terminate()
                    raise TimeoutError(f"Timed out after {timeout_s}s")
            return proc.wait(timeout=10)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        except TimeoutError:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            raise


def _load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file missing: {metrics_path}")
    with metrics_path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _print_summary(metrics: dict) -> None:
    timing = metrics.get("timing", {})
    det = metrics.get("detections", {})
    flight = metrics.get("flight", {})
    tracking = metrics.get("tracking", {})
    assess = metrics.get("assessment", {})

    print("\n" + "=" * 72)
    print("Auto Sim Summary")
    print("=" * 72)
    print(f"Duration:              {timing.get('duration_s', 0)} s")
    print(f"Camera FPS:            {timing.get('avg_capture_fps', 0)}")
    print(f"Process FPS:           {timing.get('avg_process_fps', 0)}")
    print(f"Detections by class:   {det.get('by_class', {})}")
    print(f"Locked target:         {flight.get('locked_target_id')} ({flight.get('locked_target_class')})")
    print(f"Start XY:              {flight.get('start_xy')}")
    print(f"End XY:                {flight.get('end_xy')}")
    print(f"Total XY movement:     {flight.get('total_xy_movement_m')} m")
    print(f"Distance to target:    {flight.get('distance_to_target_start_m')} -> {flight.get('distance_to_target_end_m')} m")
    print(f"Standoff error:        {flight.get('standoff_error_start_m')} -> {flight.get('standoff_error_end_m')} m")
    print(f"Pixel error:           {flight.get('pixel_error_start_px')} -> {flight.get('pixel_error_end_px')} px")
    print(f"Pixel pre-lock:        {flight.get('pixel_error_prelock_px')}")
    print(f"Offboard active:       {flight.get('offboard_active_pct')} %")
    print(f"Projection valid:      {flight.get('world_projection_valid_pct')} %")
    print(f"Target locked:         {flight.get('target_locked_pct')} %")
    print(f"Reacquisitions:        {tracking.get('reacquisitions')}")
    print(f"Assessment:            {assess.get('overall')} {assess.get('issues', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full autonomous Sentinel simulation demo")
    parser.add_argument("--simulation-only", action="store_true", help="Required: acknowledge this runner is simulation-only")
    parser.add_argument("--world", default="dynamic_v3", help="World alias or full world name")
    parser.add_argument("--topic", default=None, help="Explicit camera topic override")
    parser.add_argument("--duration", type=float, default=180.0, help="Demo duration in seconds")
    parser.add_argument("--target-fps", type=float, default=15.0, help="YOLO processing FPS")
    parser.add_argument("--takeoff-altitude", type=float, default=20.0, help="Takeoff altitude in meters")
    parser.add_argument("--standoff-altitude", type=float, default=15.0, help="Standoff altitude in meters")
    parser.add_argument("--standoff-distance", type=float, default=15.0, help="Standoff distance in meters")
    parser.add_argument(
        "--standoff-profile",
        choices=["conservative", "normal", "aggressive-sim"],
        default="aggressive-sim",
        help="Simulation standoff profile",
    )
    parser.add_argument("--spawn-pose", default="220,-350,22,0,0,0", help="PX4_GZ_MODEL_POSE for SITL spawn")
    parser.add_argument("--mavlink-url", default="udpin://0.0.0.0:14540", help="MAVSDK connection URL")
    parser.add_argument("--model-path", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--camera-timeout", type=float, default=20.0, help="Seconds to wait for camera frames")
    parser.add_argument("--stack-timeout", type=float, default=240.0, help="Max total time for the YOLO/follow run")
    parser.add_argument("--output-dir", default=None, help="Output directory for logs and metrics")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV display")
    parser.add_argument("--no-stack-launch", action="store_true", help="Assume SITL stack is already running")
    parser.add_argument("--no-land", action="store_true", help="Skip landing on shutdown, send hold only")
    args = parser.parse_args()

    _require_simulation_only(args)

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = _output_dir(args, repo_root)
    log_path = output_dir / "run.log"
    metrics_path = output_dir / "metrics.json"
    camera_frame_path = output_dir / "camera_probe.jpg"
    annotated_dir = output_dir / "annotated_frames"
    python_executable = _python_executable(repo_root)
    reporter = SimReporter(output_dir)

    launcher = SimStackLauncher(sentinel_dir=repo_root)
    world_name = launcher.normalize_world_name(args.world)
    camera_topic = _world_topic(launcher, world_name, args.topic)
    env = launcher.build_env(spawn_pose=args.spawn_pose)
    env["PYTHONUNBUFFERED"] = "1"

    print("SIMULATION ONLY — DO NOT USE WITH REAL HARDWARE")
    print(f"World:                {world_name}")
    print(f"Camera topic:         {camera_topic}")
    print(f"Output dir:           {output_dir}")
    print(f"Log file:             {log_path}")
    print()

    shutdown_result = {"hold": False, "land": False, "stack_stopped": False}
    run_cmd = _build_test_cmd(args, python_executable, camera_topic, annotated_dir)
    print("Command:")
    print("  " + shlex.join(run_cmd))

    exit_code = 0
    try:
        if not args.no_stack_launch:
            launcher.stop()
            if not launcher.start(world=world_name, spawn_pose=args.spawn_pose, no_qgc=True):
                raise RuntimeError("Failed to start PX4 SITL + Gazebo stack")

        if not launcher.wait_for_clock(world_name, timeout_s=90.0):
            raise RuntimeError("Gazebo /clock validation failed")

        if not launcher.validate_camera(camera_topic, timeout_s=args.camera_timeout, save_frame=str(camera_frame_path)):
            raise RuntimeError("Camera frame validation failed")

        # Auto-takeoff before YOLO test
        print("\n[AutoSim] Taking off...")
        import asyncio
        to_ok = asyncio.run(_auto_takeoff(args.mavlink_url, altitude_m=args.takeoff_altitude, timeout_s=60.0))
        if not to_ok:
            raise RuntimeError("Auto-takeoff failed")

        with log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"SIMULATION ONLY — DO NOT USE WITH REAL HARDWARE\n")
            log_fp.write(f"World: {world_name}\n")
            log_fp.write(f"Camera topic: {camera_topic}\n")
            log_fp.write(f"Started: {datetime.now(timezone.utc).isoformat()}\n")

        rc = _stream_subprocess(run_cmd, cwd=repo_root, env=env, log_path=log_path, timeout_s=args.stack_timeout)
        if rc != 0:
            exit_code = rc
            raise RuntimeError(f"YOLO/follow run exited with status {rc}")

    except KeyboardInterrupt:
        exit_code = 130
        print("\nInterrupted, beginning safe shutdown...")
    except Exception as exc:
        if exit_code == 0:
            exit_code = 1
        print(f"\nAUTO SIM FAILED: {exc}", file=sys.stderr)
    finally:
        if not args.no_stack_launch:
            launcher.stop()
            shutdown_result["stack_stopped"] = True

    # NOTE: run_gazebo_yolo_test.py does not produce metrics.json.
    # We treat a clean exit as success and parse the log for summary stats.
    if log_path.exists() and exit_code == 0:
        print(f"\nRun log: {log_path}")
        print(f"Annotated frames: {annotated_dir}")
        print(f"Camera probe: {camera_frame_path}")
    elif exit_code == 0:
        print("Run completed but no log file produced.", file=sys.stderr)
        exit_code = 1

    print("\nShutdown result:")
    print(json.dumps(shutdown_result, indent=2))
    print(f"Artifacts:            {output_dir}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
