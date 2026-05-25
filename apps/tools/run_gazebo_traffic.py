"""Move vehicle models along predefined routes via Gazebo set_pose service.

Usage:
    # With default sentinel_street routes:
    python -m apps.tools.run_gazebo_traffic --world sentinel_street

    # Custom update rate:
    python -m apps.tools.run_gazebo_traffic --world sentinel_street --hz 10

    # Dry-run (no service calls, just print trajectories):
    python -m apps.tools.run_gazebo_traffic --world sentinel_street --dry-run

Prerequisites:
    - Gazebo running with the target world loaded
    - gz.transport13 / gz.msgs10 Python packages installed
    - PX4's UserCommands plugin loaded (via server.config)
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import threading
from dataclasses import dataclass

try:
    import gz.transport13 as _gzt
    import gz.msgs10.pose_pb2 as _pose_pb2
    import gz.msgs10.boolean_pb2 as _bool_pb2

    _GZ_AVAILABLE = True
except ImportError:
    _GZ_AVAILABLE = False


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float
    speed: float  # m/s to next waypoint


@dataclass
class Route:
    model_name: str
    waypoints: list[Waypoint]
    loop: bool = True
    yaw_offset: float = 0.0  # compensation for Fuel model internal mesh rotation


# Fuel models have internal mesh yaw offsets that must be compensated
# so the vehicle faces the correct world direction.
#
# Derivation (from actual cached model.sdf files):
#   internal_yaw = direction the mesh faces in the link frame when yaw=0
#   The model SDF visual <pose> rotation β is applied to the raw mesh forward α:
#     internal_yaw = α + β
#   yaw_offset = -internal_yaw  (compensates so effective_yaw aligns mesh with heading)
#
#   Hatchback: raw mesh forward=-Y (α=-π/2), model SDF visual rotation=+π/2 (β=+π/2)
#     → internal_yaw = -π/2 + π/2 = 0, offset = 0
#   Pickup: raw mesh forward=+Y (α=+π/2), model SDF visual rotation=-π/2 (β=-π/2)
#     → internal_yaw = π/2 - π/2 = 0, offset = 0
#   Prius Hybrid: collision boxes show front at -Y (α=-π/2), no visual rotation (β=0)
#     → internal_yaw = -π/2, offset = +π/2
#   TruckBox: mesh length along +X (α=0), no visual rotation (β=0)
#     → internal_yaw = 0, offset = 0
#   Bus: mesh length along +Y (α=+π/2), no visual rotation (β=0)
#     → internal_yaw = +π/2, offset = -π/2
FUEL_YAW_OFFSETS: dict[str, float] = {
    "vehicle_moving_1": 0.0,              # Pickup — model SDF -π/2 reorients +Y→+X, faces +X at yaw=0
    "vehicle_moving_2": 0.0,              # Hatchback — model SDF +π/2 reorients -Y→+X, faces +X at yaw=0
    "vehicle_moving_3": math.pi / 2,      # Prius — front at -Y per collision boxes, faces -Y at yaw=0
    "vehicle_moving_4": 0.0,              # TruckBox — mesh forward along +X, no rotation needed
}


# Density → max number of routes (models must exist in the SDF).
DENSITY_ROUTE_COUNT: dict[str, int] = {
    "light": 1,
    "medium": 2,
    "heavy": 4,
}


def _all_routes() -> list[Route]:
    """All sentinel_street vehicle routes (heavy density = 4 vehicles)."""
    return [
        # Pickup: east along south lane (urban speeds 2-5 m/s)
        Route(
            model_name="vehicle_moving_1",
            yaw_offset=FUEL_YAW_OFFSETS["vehicle_moving_1"],
            waypoints=[
                Waypoint(x=-50, y=-2.5, z=0, yaw=0, speed=4),
                Waypoint(x=-20, y=-2.5, z=0, yaw=0, speed=5),
                Waypoint(x=5, y=-2.5, z=0, yaw=0, speed=2),    # slow near intersection
                Waypoint(x=15, y=-2.5, z=0, yaw=0, speed=4),
                Waypoint(x=50, y=-2.5, z=0, yaw=0, speed=5),
            ],
        ),
        # Hatchback: west along north lane (urban speeds 2-5 m/s)
        Route(
            model_name="vehicle_moving_2",
            yaw_offset=FUEL_YAW_OFFSETS["vehicle_moving_2"],
            waypoints=[
                Waypoint(x=50, y=2.5, z=0, yaw=0, speed=4),
                Waypoint(x=20, y=2.5, z=0, yaw=0, speed=5),
                Waypoint(x=-5, y=2.5, z=0, yaw=0, speed=2),    # slow near intersection
                Waypoint(x=-15, y=2.5, z=0, yaw=0, speed=4),
                Waypoint(x=-50, y=2.5, z=0, yaw=0, speed=5),
            ],
        ),
        # Prius: slow east along south outer lane (2-3 m/s)
        Route(
            model_name="vehicle_moving_3",
            yaw_offset=FUEL_YAW_OFFSETS["vehicle_moving_3"],
            waypoints=[
                Waypoint(x=-40, y=-3.5, z=0, yaw=0, speed=2),
                Waypoint(x=-10, y=-3.5, z=0, yaw=0, speed=3),
                Waypoint(x=0, y=-3.5, z=0, yaw=0, speed=2),    # crosswalk
                Waypoint(x=5, y=-3.5, z=0, yaw=0, speed=3),
                Waypoint(x=20, y=-3.5, z=0, yaw=0, speed=3),
                Waypoint(x=40, y=-3.5, z=0, yaw=0, speed=2),
            ],
        ),
        # TruckBox: west along north outer lane (heavy, 2-4 m/s)
        Route(
            model_name="vehicle_moving_4",
            yaw_offset=FUEL_YAW_OFFSETS["vehicle_moving_4"],
            waypoints=[
                Waypoint(x=45, y=3.5, z=0, yaw=0, speed=3),
                Waypoint(x=20, y=3.5, z=0, yaw=0, speed=4),
                Waypoint(x=0, y=3.5, z=0, yaw=0, speed=2),     # intersection caution
                Waypoint(x=-20, y=3.5, z=0, yaw=0, speed=3),
                Waypoint(x=-45, y=3.5, z=0, yaw=0, speed=2),
            ],
        ),
    ]


def _default_routes(density: str = "medium") -> list[Route]:
    """Return routes filtered by density level."""
    count = DENSITY_ROUTE_COUNT.get(density, 2)
    return _all_routes()[:count]


def _interpolate_route(route: Route, elapsed: float) -> tuple[float, float, float, float] | None:
    """Compute interpolated (x, y, z, yaw) along a route at given elapsed time."""
    wps = route.waypoints
    if len(wps) < 2:
        return None

    # Compute cumulative segment times
    seg_times: list[float] = [0.0]
    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        dist = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        speed = a.speed if a.speed > 0 else 10.0
        seg_times.append(seg_times[-1] + dist / speed)

    total_time = seg_times[-1]
    if total_time <= 0:
        return wps[0].x, wps[0].y, wps[0].z, wps[0].yaw

    t = elapsed % total_time if route.loop else min(elapsed, total_time)

    for i in range(len(seg_times) - 1):
        if seg_times[i] <= t <= seg_times[i + 1]:
            seg_duration = seg_times[i + 1] - seg_times[i]
            frac = (t - seg_times[i]) / seg_duration if seg_duration > 0 else 0.0
            a, b = wps[i], wps[i + 1]
            x = a.x + (b.x - a.x) * frac
            y = a.y + (b.y - a.y) * frac
            z = a.z + (b.z - a.z) * frac
            # Compute yaw from velocity direction, not waypoint yaw values
            dx = b.x - a.x
            dy = b.y - a.y
            yaw = math.atan2(dy, dx) if abs(dx) > 0.01 or abs(dy) > 0.01 else a.yaw
            return x, y, z, yaw

    return wps[-1].x, wps[-1].y, wps[-1].z, wps[-1].yaw


def _make_pose_msg(model_name: str, x: float, y: float, z: float, yaw: float) -> object:
    """Build a gz.msgs.Pose message."""
    msg = _pose_pb2.Pose()
    msg.name = model_name
    msg.position.x = x
    msg.position.y = y
    msg.position.z = z

    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    msg.orientation.x = 0.0
    msg.orientation.y = 0.0
    msg.orientation.z = qz
    msg.orientation.w = qw

    return msg


def run_traffic(world: str, routes: list[Route], hz: float, dry_run: bool) -> None:
    """Main traffic loop."""
    service = f"/world/{world}/set_pose"
    node = _gzt.Node()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    print(f"Traffic controller: {len(routes)} vehicles")
    print(f"  world: {world}")
    print(f"  service: {service}")
    print(f"  rate: {hz} Hz")
    print(f"  dry_run: {dry_run}")
    print()

    for r in routes:
        print(f"  {r.model_name}: {len(r.waypoints)} waypoints")
    print()

    t0 = time.monotonic()
    update_count = 0
    last_report = t0

    while not stop.is_set():
        elapsed = time.monotonic() - t0

        for route in routes:
            result = _interpolate_route(route, elapsed)
            if result is None:
                continue
            x, y, z, yaw = result

            if dry_run:
                if update_count == 0:
                    print(f"  [DRY] {route.model_name}: ({x:.1f}, {y:.1f}, {z:.1f}) yaw={yaw:.2f}")
            else:
                effective_yaw = yaw + route.yaw_offset
                msg = _make_pose_msg(route.model_name, x, y, z, effective_yaw)
                ok, _ = node.request(
                    service, msg, _pose_pb2.Pose, _bool_pb2.Boolean, timeout=100,
                )
                if not ok and update_count < 5:
                    print(f"  WARN: set_pose failed for {route.model_name}")

        update_count += 1

        now = time.monotonic()
        if now - last_report >= 5.0:
            rps = update_count / (now - t0)
            print(f"  [{elapsed:.0f}s] updates: {update_count}  rate: {rps:.1f}/s")
            last_report = now

        time.sleep(1.0 / hz)

    print(f"\nStopped after {update_count} updates in {time.monotonic() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move Gazebo vehicle models along routes")
    parser.add_argument("--world", default="sentinel_street", help="Gazebo world name")
    parser.add_argument("--hz", type=float, default=20.0, help="Pose update rate (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Print poses without sending")
    parser.add_argument(
        "--density", default="medium", choices=["light", "medium", "heavy"],
        help="Scene density: light=1 vehicle, medium=2, heavy=4 (default: medium)",
    )
    args = parser.parse_args()

    if not _GZ_AVAILABLE:
        print("ERROR: gz.transport13 / gz.msgs10 not available", file=sys.stderr)
        sys.exit(1)

    routes = _default_routes(args.density)
    run_traffic(args.world, routes, args.hz, args.dry_run)


if __name__ == "__main__":
    main()
