"""Gazebo performance report — RTF, camera FPS, topic audit, density check.

Checks:
  1. Active scene density from sentinel_street.sdf header comment
  2. Real-Time Factor from /stats topic
  3. x500_mono_cam camera FPS
  4. Extra vehicle camera/laser/sonar topics (performance drain)
  5. Recommended scene density based on RTF

Usage:
    python -m apps.tools.gazebo_perf_report
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path


SDF_PATH = Path("configs/gz/sentinel_street.sdf")


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_topic_list() -> list[str]:
    out = _run(["gz", "topic", "-l"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def _get_topic_rate(topic: str, seconds: float = 3) -> float:
    cmd = ["gz", "topic", "-e", "-t", topic, "-n", "10"]
    try:
        start = time.monotonic()
        subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 5)
        elapsed = time.monotonic() - start
        return min(10.0 / max(elapsed, 0.1), 999.0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0


def _detect_density() -> str:
    """Read density marker from sentinel_street.sdf header."""
    if not SDF_PATH.exists():
        return "unknown"
    try:
        header = SDF_PATH.read_text().split("\n", 5)
        for line in header:
            m = re.search(r"SCENE_DENSITY:\s*(light|medium|heavy)", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return "unknown"


def main() -> None:
    print("Gazebo Performance Report")
    print("=" * 50)
    print()

    # Detect active density
    density = _detect_density()
    sdf_resolved = SDF_PATH.resolve()
    print(f"Active density: {density}")
    print(f"SDF file: {SDF_PATH} -> {sdf_resolved.name}")
    print()

    topics = _get_topic_list()
    if not topics:
        print("ERROR: No Gazebo topics found. Is Gazebo running?")
        print("  Start with: make px4-sentinel-world-clean")
        print("  Or:         SCENE_DENSITY=light make px4-sentinel-world-clean")
        sys.exit(1)

    print(f"Total topics: {len(topics)}")
    print()

    # Camera topic detection
    camera_topics = [t for t in topics if "/camera/" in t and "/image" in t]
    vehicle_sensor_topics = [
        t for t in topics
        if any(kw in t.lower() for kw in ("vehicle_moving", "prius", "hatchback", "pickup", "truckbox", "bus"))
        and any(sensor in t.lower() for sensor in ("/camera/", "/lidar/", "/gpu_lidar/", "/sonar/", "/imu/", "/gps"))
    ]

    # x500 camera FPS
    x500_cam_topics = [t for t in camera_topics if "x500" in t]
    x500_fps = 0.0
    if x500_cam_topics:
        print(f"x500 camera topic: {x500_cam_topics[0]}")
        x500_fps = _get_topic_rate(x500_cam_topics[0], seconds=3)
        print(f"  Measured FPS: {x500_fps:.1f}")
    else:
        print("WARN: No x500 camera topic found")
    print()

    # Vehicle sensor topics
    print("Vehicle sensor topics:")
    if vehicle_sensor_topics:
        for t in vehicle_sensor_topics:
            print(f"  WARN: {t}")
        print(f"  Total: {len(vehicle_sensor_topics)} — these drain performance")
    else:
        print("  OK: No vehicle sensor topics (visual-only models)")
    print()

    # RTF
    print("Real-Time Factor:")
    rtf = None
    stats_topics = [t for t in topics if t.endswith("/stats")]
    if stats_topics:
        out = _run(["gz", "topic", "-e", "-t", stats_topics[0], "-n", "5"], timeout=8)
        if out:
            for line in out.splitlines():
                if "real_time_factor" in line.lower():
                    m = re.search(r"(\d+\.?\d*)", line)
                    if m:
                        rtf = float(m.group(1))
                        break
    if rtf is not None:
        print(f"  RTF: {rtf:.3f}")
    else:
        print("  (could not measure RTF — check gz gui)")
    print()

    # Summary
    print("=" * 50)
    print("Recommendation:")
    issues = []
    if vehicle_sensor_topics:
        issues.append("Vehicle sensor topics detected")
        print("  Switch to MEDIUM or LIGHT density to remove sensor-heavy vehicles")
        print("  (Prius Hybrid with sensors creates ~15 extra topics)")
    if x500_fps > 0 and x500_fps < 20:
        issues.append(f"Camera FPS {x500_fps:.1f} < 20")
        print(f"  Camera FPS ({x500_fps:.1f}) below 20 — consider LIGHT density")
    if rtf is not None and rtf < 0.9:
        issues.append(f"RTF {rtf:.3f} < 0.9")
        print(f"  RTF ({rtf:.3f}) below 0.9 — consider reducing scene density")
        print("  Try: SCENE_DENSITY=light make px4-sentinel-world-clean")
        print("  Or:  make px4-sentinel-world-light")
    if not issues:
        print("  Performance OK — scene density is appropriate")


if __name__ == "__main__":
    main()
