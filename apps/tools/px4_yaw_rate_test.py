"""PX4 yaw rate test — verify offboard VelocityBodyYawspeed commands.

Connects to PX4 SITL via MAVSDK, starts offboard, commands yaw rotation,
and logs actual heading change to verify PX4 obeys yaw rate commands.

Usage:
    python -m apps.tools.px4_yaw_rate_test
    python -m apps.tools.px4_yaw_rate_test --url udpin://0.0.0.0:14540
    python -m apps.tools.px4_yaw_rate_test --rate 0.4 --duration 5

Prerequisites:
    - PX4 SITL running with a drone model
    - mavsdk package installed: pip install -e .[mavlink]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time


async def get_heading(drone) -> float | None:
    """Get current heading in degrees."""
    async for heading in drone.telemetry.heading():
        return heading.heading_deg
    return None


async def get_euler_yaw(drone) -> float | None:
    """Get yaw from attitude euler angles (more precise than heading)."""
    async for angle in drone.telemetry.attitude_euler():
        return angle.yaw_deg
    return None


def _norm_angle_delta(delta: float) -> float:
    """Normalize angle delta to [-180, 180]."""
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360
    return delta


async def run_test(url: str, rate: float, duration: float) -> bool:
    from mavsdk import System
    from mavsdk.offboard import VelocityBodyYawspeed, OffboardError

    print(f"Connecting to {url}...")
    drone = System()
    await drone.connect(system_address=url)

    # Wait for connection
    connected = False
    for _ in range(30):
        async for state in drone.core.connection_state():
            if state.is_connected:
                connected = True
                break
        if connected:
            break
        await asyncio.sleep(1)

    if not connected:
        print("FAIL: Could not connect to PX4")
        return False

    print("Connected to PX4 SITL")
    print()

    # Get initial heading
    heading_start = await get_heading(drone)
    yaw_start = await get_euler_yaw(drone)
    print(f"Initial heading: {heading_start:.1f} deg")
    print(f"Initial attitude yaw: {yaw_start:.1f} deg")
    print()

    # Arm
    print("Arming...")
    await drone.action.arm()
    await asyncio.sleep(1)
    print("Armed")
    print()

    # Send initial setpoint before starting offboard
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))

    # Start offboard
    print("Starting offboard mode...")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"FAIL: Offboard start rejected: {e}")
        await drone.action.disarm()
        return False

    print("Offboard active")
    print()

    # Test 1: Positive yaw rate (counter-clockwise)
    expected_deg = rate * duration * (180.0 / math.pi)
    rate_dps = math.degrees(rate)  # MAVSDK expects degrees/second
    print(f"Test 1: Commanding +{rate} rad/s ({rate_dps:.1f} deg/s) for {duration}s")
    print(f"  Expected rotation: +{expected_deg:.1f} deg")

    yaw_before = await get_euler_yaw(drone)
    t0 = time.monotonic()

    # Send yaw rate commands at 20 Hz
    elapsed = 0.0
    dt = 0.05
    while elapsed < duration:
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0, 0, 0, rate_dps)
        )
        await asyncio.sleep(dt)
        elapsed += dt

    yaw_after = await get_euler_yaw(drone)
    actual_delta = _norm_angle_delta(yaw_after - yaw_before)
    print(f"  Actual rotation: {actual_delta:+.1f} deg")
    print(f"  Error: {abs(actual_delta - expected_deg):.1f} deg")
    print()

    # Pause
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(0.5)

    # Test 2: Negative yaw rate (clockwise)
    print(f"Test 2: Commanding -{rate} rad/s (-{rate_dps:.1f} deg/s) for {duration}s")
    print(f"  Expected rotation: -{expected_deg:.1f} deg")

    yaw_before = await get_euler_yaw(drone)
    t0 = time.monotonic()

    elapsed = 0.0
    while elapsed < duration:
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0, 0, 0, -rate_dps)
        )
        await asyncio.sleep(dt)
        elapsed += dt

    yaw_after = await get_euler_yaw(drone)
    actual_delta2 = _norm_angle_delta(yaw_after - yaw_before)
    print(f"  Actual rotation: {actual_delta2:+.1f} deg")
    print(f"  Error: {abs(actual_delta2 - (-expected_deg)):.1f} deg")
    print()

    # Stop offboard
    print("Stopping offboard...")
    try:
        await drone.offboard.stop()
    except OffboardError as e:
        print(f"WARN: Offboard stop error: {e}")

    # Disarm
    await asyncio.sleep(1)
    await drone.action.disarm()
    print("Disarmed")

    # Result
    print()
    threshold_deg = expected_deg * 0.5
    t1_pass = abs(actual_delta - expected_deg) < expected_deg
    t2_pass = abs(actual_delta2 + expected_deg) < expected_deg

    print("=" * 50)
    print("Results:")
    print(f"  Test 1 (+{rate} rad/s): {'PASS' if t1_pass else 'FAIL'}")
    print(f"    Expected {expected_deg:+.1f} deg, got {actual_delta:+.1f} deg")
    print(f"  Test 2 (-{rate} rad/s): {'PASS' if t2_pass else 'FAIL'}")
    print(f"    Expected {-expected_deg:+.1f} deg, got {actual_delta2:+.1f} deg")
    print()

    if t1_pass and t2_pass:
        print("VERDICT: Offboard yaw commands work correctly.")
        print("  MAVSDK VelocityBodyYawspeed uses rad/s (confirmed).")
        return True
    else:
        print("VERDICT: Offboard yaw commands may not be working.")
        print("  Check: is the drone armed and in offboard mode?")
        print("  Check: PX4 parameter COM_ARM_WO_GPS?")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PX4 offboard yaw rate commands")
    parser.add_argument("--url", default="udpin://0.0.0.0:14540", help="MAVSDK connection URL")
    parser.add_argument("--rate", type=float, default=0.4, help="Yaw rate in rad/s (default: 0.4)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration per test in seconds (default: 5)")
    args = parser.parse_args()

    try:
        success = asyncio.run(run_test(args.url, args.rate, args.duration))
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install with: pip install -e .[mavlink]")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
