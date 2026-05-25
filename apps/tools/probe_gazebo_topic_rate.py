"""Probe Gazebo topic message rate and simulation status.

Usage:
    # Auto-detect camera topic and measure rate:
    python -m apps.tools.probe_gazebo_topic_rate --auto-topic --seconds 5

    # Probe a specific topic:
    python -m apps.tools.probe_gazebo_topic_rate \\
        --topic /world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image \\
        --seconds 5

    # Check simulation health (auto-detects world):
    python -m apps.tools.probe_gazebo_topic_rate --stats --seconds 5

    # Check specific world stats:
    python -m apps.tools.probe_gazebo_topic_rate --stats --world sentinel_street --seconds 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

try:
    import gz.transport13 as _gzt
    import gz.msgs10.image_pb2 as _image_pb2
    import gz.msgs10.imu_pb2 as _imu_pb2
    import gz.msgs10.fluid_pressure_pb2 as _fluid_pb2
    import gz.msgs10.magnetometer_pb2 as _mag_pb2
    import gz.msgs10.navsat_pb2 as _navsat_pb2
    import gz.msgs10.world_stats_pb2 as _stats_pb2

    _GZ_AVAILABLE = True
except ImportError:
    _GZ_AVAILABLE = False

_MSG_TYPE_MAP = {
    "image": ("Image", _image_pb2.Image),
    "imu": ("IMU", _imu_pb2.IMU),
    "air_pressure": ("FluidPressure", _fluid_pb2.FluidPressure),
    "magnetometer": ("Magnetometer", _mag_pb2.Magnetometer),
    "navsat": ("NavSat", _navsat_pb2.NavSat),
}


def _discover_worlds() -> list[str]:
    """Discover running Gazebo worlds by listing clock topics."""
    try:
        result = subprocess.run(
            ["gz", "topic", "-l"],
            capture_output=True, text=True, timeout=5,
        )
        topics = result.stdout.strip().split("\n") if result.stdout.strip() else []
        worlds = []
        for t in topics:
            t = t.strip()
            if t.startswith("/world/") and t.endswith("/clock"):
                name = t[len("/world/"):-len("/clock")]
                worlds.append(name)
        return worlds
    except Exception:
        return []


def _discover_camera_topics() -> list[str]:
    """Discover camera image topics, preferring drone (x500_mono_cam) over vehicle cameras."""
    # Allow env var override
    env_topic = os.environ.get("DRONE_CAMERA_TOPIC")
    if env_topic:
        return [env_topic]

    try:
        result = subprocess.run(
            ["gz", "topic", "-l"],
            capture_output=True, text=True, timeout=5,
        )
        topics = result.stdout.strip().split("\n") if result.stdout.strip() else []
        camera_topics = [t.strip() for t in topics if "camera" in t and "image" in t and "/sensor/" in t]
    except Exception:
        return []

    # Prefer drone camera topics
    drone_topics = [t for t in camera_topics if "x500_mono_cam" in t or "mono_cam_0" in t]
    if drone_topics:
        return drone_topics

    return camera_topics


def _detect_msg_type(topic: str):
    for hint, (name, msg_type) in _MSG_TYPE_MAP.items():
        if hint in topic:
            return name, msg_type
    return "Image", _image_pb2.Image


def _check_single_world(worlds: list[str]) -> str | None:
    """Return world name if exactly one running, else print diagnostics and return None."""
    if not worlds:
        print("FAIL: no Gazebo worlds detected. Is Gazebo running?")
        return None
    if len(worlds) == 1:
        return worlds[0]
    print(f"FAIL: {len(worlds)} Gazebo worlds detected:")
    for w in worlds:
        print(f"  - {w}")
    print("Kill stale instances: make gazebo-kill-stale")
    return None


def _probe_topic(topic: str, duration_s: float) -> None:
    type_name, msg_type = _detect_msg_type(topic)

    count = 0
    lock = threading.Lock()
    first_ts: list[float] = []
    last_ts: list[float] = []

    def _on_msg(msg: object) -> None:
        nonlocal count
        with lock:
            count += 1
            now = time.monotonic()
            if not first_ts:
                first_ts.append(now)
            last_ts.clear()
            last_ts.append(now)

    node = _gzt.Node()
    node.subscribe(msg_type=msg_type, topic=topic, callback=_on_msg)

    print(f"Probing: {topic}")
    print(f"  msg_type: {type_name}")
    print(f"  duration: {duration_s}s")
    print()

    t0 = time.monotonic()
    per_second_count = 0
    per_second_start = t0

    while time.monotonic() - t0 < duration_s:
        time.sleep(0.1)
        elapsed = time.monotonic() - t0
        with lock:
            c = count

        sec_elapsed = time.monotonic() - per_second_start
        if sec_elapsed >= 1.0:
            rate = (c - per_second_count) / sec_elapsed
            print(f"  [{elapsed:5.1f}s] total: {c}  current: {rate:.1f} Hz")
            per_second_count = c
            per_second_start = time.monotonic()

    with lock:
        total = count

    elapsed = time.monotonic() - t0
    avg_hz = total / elapsed if elapsed > 0 else 0

    print()
    print(f"Result: {total} messages in {elapsed:.1f}s = {avg_hz:.1f} Hz")

    with lock:
        if first_ts and last_ts:
            span = last_ts[0] - first_ts[0]
            if total > 1 and span > 0:
                print(f"  inter-message rate: {(total - 1) / span:.1f} Hz")

    if total == 1:
        print("  FAIL: only 1 frame received — camera rendering is stalled")
        print("  Likely causes:")
        print("    1. Sensors system not loaded (check GZ_SIM_SERVER_CONFIG_PATH)")
        print("    2. ogre2 render context failed (check GPU/display)")
        print("    3. Multiple Gazebo instances (run: make gazebo-kill-stale)")
    elif avg_hz < 1.0:
        print(f"  WARNING: rate {avg_hz:.1f} Hz is below 1 Hz — rendering degraded")
    elif avg_hz < 5.0:
        print(f"  WARNING: rate {avg_hz:.1f} Hz is below 5 Hz — may be insufficient for detection")
    else:
        print(f"  OK: {avg_hz:.1f} Hz is above 5 Hz threshold")


def _probe_stats(world: str, duration_s: float) -> None:
    stats_topic = f"/world/{world}/stats"
    latest: dict = {}
    lock = threading.Lock()
    recv_count = 0

    def _on_stats(msg: object) -> None:
        nonlocal recv_count
        with lock:
            recv_count += 1
            latest["sim_time_sec"] = msg.sim_time.sec + msg.sim_time.nsec * 1e-9
            latest["real_time_sec"] = msg.real_time.sec + msg.real_time.nsec * 1e-9
            latest["paused"] = msg.paused
            latest["real_time_factor"] = msg.real_time_factor
            latest["iterations"] = msg.iterations
            latest["step_size"] = msg.step_size
            latest["model_count"] = msg.model_count

    node = _gzt.Node()
    node.subscribe(msg_type=_stats_pb2.WorldStatistics, topic=stats_topic, callback=_on_stats)

    print(f"Probing world stats: {stats_topic}")
    print(f"  duration: {duration_s}s")
    print()

    t0 = time.monotonic()
    prev_sim_time: float | None = None
    prev_iterations: int | None = None
    monotonic_errors = 0

    while time.monotonic() - t0 < duration_s:
        time.sleep(1.0)
        with lock:
            s = dict(latest)
            rc = recv_count

        if not s:
            print("  No stats received — is Gazebo running?")
            continue

        sim_t = s.get("sim_time_sec", 0)
        iters = s.get("iterations", 0)
        paused = s.get("paused", False)
        rtf = s.get("real_time_factor", 0)
        models = s.get("model_count", 0)
        step = s.get("step_size", 0)

        # Check monotonicity
        if prev_sim_time is not None and sim_t < prev_sim_time:
            monotonic_errors += 1

        delta_str = ""
        if prev_sim_time is not None:
            sim_delta = sim_t - prev_sim_time
            iter_delta = iters - (prev_iterations or 0)
            delta_str = f"  Δsim: {sim_delta:+.3f}s  Δiter: {iter_delta:+d}"
            if sim_delta < -0.001:
                delta_str += "  ** JUMPED BACK **"

        print(
            f"  [{time.monotonic() - t0:5.1f}s] "
            f"sim: {sim_t:.3f}s{delta_str}  "
            f"paused: {paused}  RTF: {rtf:.3f}  "
            f"models: {models}  step: {step}  stats_msgs: {rc}"
        )

        if paused:
            print("  ** SIMULATION IS PAUSED **")

        prev_sim_time = sim_t
        prev_iterations = iters

    print()
    with lock:
        s = dict(latest)

    ok = True
    if not s:
        print("FAIL: no stats received on", stats_topic)
        ok = False
    if s.get("paused"):
        print("FAIL: simulation is PAUSED — start with -r flag")
        ok = False
    if monotonic_errors > 0:
        print(f"FAIL: sim_time jumped backwards {monotonic_errors} time(s) — stale/multiple world instances")
        ok = False
    if s.get("model_count", -1) == 0:
        print("WARN: model_count is 0 — model may not have spawned yet")
    if ok and s:
        print(f"OK: simulation running (RTF: {s.get('real_time_factor', 0):.3f}, models: {s.get('model_count', 0)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Gazebo topic rate and simulation status")
    parser.add_argument("--topic", help="Gazebo topic to probe")
    parser.add_argument("--auto-topic", action="store_true", help="Auto-discover camera topic")
    parser.add_argument("--seconds", type=float, default=5.0, help="Probe duration (default: 5)")
    parser.add_argument("--msg-type", choices=list(_MSG_TYPE_MAP.keys()), help="Force message type")
    parser.add_argument("--stats", action="store_true", help="Probe world stats (sim_time, paused, RTF)")
    parser.add_argument("--world", default="", help="World name for --stats (auto-detected if omitted)")
    args = parser.parse_args()

    if not _GZ_AVAILABLE:
        print("ERROR: gz.transport13 / gz.msgs10 not available", file=sys.stderr)
        sys.exit(1)

    if args.stats:
        worlds = _discover_worlds()
        world = args.world or ""
        if not world:
            w = _check_single_world(worlds)
            if w is None:
                sys.exit(1)
            world = w
        elif world not in worlds:
            print(f"FAIL: world '{world}' not found. Running: {worlds or 'none'}")
            sys.exit(1)
        _probe_stats(world, args.seconds)

    elif args.auto_topic:
        env_topic = os.environ.get("DRONE_CAMERA_TOPIC")
        if env_topic:
            print(f"Using DRONE_CAMERA_TOPIC env: {env_topic}")
            _probe_topic(env_topic, args.seconds)
            return

        topics = _discover_camera_topics()
        if not topics:
            print("FAIL: no camera image topics found. Is PX4 running with x500_mono_cam?")
            sys.exit(1)

        selected = topics[0]
        if len(topics) > 1:
            # Prefer drone camera
            drone = [t for t in topics if "x500_mono_cam" in t or "mono_cam_0" in t]
            if drone:
                selected = drone[0]
                print(f"Found {len(topics)} camera topics, preferring drone camera:")
            else:
                print(f"WARNING: no x500_mono_cam topic found among {len(topics)} camera topics.")
                print(f"  Set DRONE_CAMERA_TOPIC to override.")
                print(f"Using first:")
            for t in topics:
                marker = " <-- selected" if t == selected else ""
                print(f"  - {t}{marker}")
        _probe_topic(selected, args.seconds)

    elif args.topic:
        _probe_topic(args.topic, args.seconds)

    else:
        parser.error("Specify --topic, --auto-topic, or --stats")


if __name__ == "__main__":
    main()
