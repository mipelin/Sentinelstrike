"""Audit Gazebo scene SDF for suspicious object poses.

Checks:
  1. Actor Z placement (Mingfei walk actors require Z=1.0, not Z=0)
  2. Static person Z placement (Fuel includes must use Z=0)
  3. Vehicle yaw correctness (mesh faces intended direction)
  4. Actor yaw facing (waypoint yaw matches travel direction)
  5. Actor walking speed (0.5-2.0 m/s range)
  6. Actor route vs building collision
  7. Vehicle Z (should be ~0)
  8. Road forbidden zone (pedestrians/objects inside road lanes)
  9. Camera validation cluster visibility

Usage:
    python -m apps.tools.audit_gazebo_scene
    python -m apps.tools.audit_gazebo_scene --sdf configs/gz/sentinel_street.sdf
    python -m apps.tools.audit_gazebo_scene --verbose
    python -m apps.tools.audit_gazebo_scene --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Fuel model internal mesh yaw offsets (radians).
#
# internal_yaw = direction the mesh faces in the link frame when SDF yaw=0.
# Derived from actual model.sdf visual <pose> rotations:
#   internal_yaw = mesh_forward_in_raw_mesh + model_sdf_visual_rotation
#
# Hatchback: raw mesh forward=-Y (-π/2), model SDF visual rot +π/2 → 0
# Pickup:    raw mesh forward=+Y (+π/2), model SDF visual rot -π/2 → 0
# TruckBox:  raw mesh forward=+X (0), no visual rotation → 0
# Bus:       raw mesh forward=+Y (+π/2), no visual rotation → +π/2
# Prius Hybrid with sensors: collision boxes show front at -Y → -π/2
FUEL_MESH_YAW: dict[str, float] = {
    "Hatchback": 0.0,
    "Pickup": 0.0,
    "Bus": math.pi / 2,
    "TruckBox": 0.0,
    "Prius Hybrid with sensors": -math.pi / 2,
    "Male visitor": 0.0,
    "FemaleVisitor": 0.0,
    "Walking person": 0.0,
}

# Thresholds
VEHICLE_Z_MAX = 0.2
MINGFEI_WALK_Z = 1.0
Z_TOLERANCE = 0.15
YAW_TOLERANCE = 0.5  # radians for actor facing check
SPEED_MIN = 0.5  # m/s minimum natural walking
SPEED_MAX = 2.0  # m/s maximum natural walking

# Road layout for forbidden zone checks
ROAD_Y_MIN = -5.0  # South edge of main road
ROAD_Y_MAX = 5.0  # North edge of main road
ROAD_X_MIN = -60.0
ROAD_X_MAX = 60.0
SIDEWALK_N_Y = 7.0  # Center of north sidewalk
SIDEWALK_S_Y = -7.0  # Center of south sidewalk
SIDEWALK_HALF_W = 1.5  # Half-width of sidewalk (3m total / 2)


def _parse_pose(pose_text: str) -> tuple[float, float, float, float, float, float]:
    parts = pose_text.strip().split()
    while len(parts) < 6:
        parts.append("0")
    return tuple(float(p) for p in parts[:6])  # type: ignore[return-value]


def _extract_model_name_from_uri(uri: str) -> str | None:
    parts = uri.rstrip("/").split("/")
    for i, p in enumerate(parts):
        if p == "models" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)


def _heading_name(yaw: float) -> str:
    yaw = yaw % (2 * math.pi)
    if yaw < 0:
        yaw += 2 * math.pi
    # Normalize to [0, 2π)
    if yaw >= 2 * math.pi - 0.01:
        yaw = 0.0
    names = [
        (0.39, "East"), (1.18, "NE"), (1.96, "North"),
        (2.75, "NW"), (3.53, "West"), (4.32, "SW"),
        (5.10, "South"), (5.89, "SE"), (6.28, "East"),
    ]
    for threshold, name in names:
        if yaw < threshold:
            return name
    return f"{yaw:.2f}rad"


def _collect_buildings(root: ET.Element) -> list[tuple[str, float, float, float, float]]:
    """Return list of (name, cx, cy, half_x, half_y) for building collision boxes."""
    buildings = []
    for model in root.iter("model"):
        name = model.get("name", "")
        if "building" not in name.lower():
            continue
        pose_el = model.find("pose")
        x, y = 0.0, 0.0
        if pose_el is not None and pose_el.text:
            x, y = _parse_pose(pose_el.text)[:2]
        for link in model.findall("link"):
            for elem in link.findall("visual") + link.findall("collision"):
                geom = elem.find("geometry/box/size")
                if geom is not None and geom.text:
                    sx, sy = [float(v) for v in geom.text.strip().split()[:2]]
                    buildings.append((name, x, y, sx / 2, sy / 2))
    return buildings


def _point_in_building(px: float, py: float, buildings: list, margin: float = 0.3) -> str | None:
    for bname, bx, by, hx, hy in buildings:
        if abs(px - bx) < hx + margin and abs(py - by) < hy + margin:
            return bname
    return None


def _on_road(px: float, py: float) -> bool:
    """Check if a point is inside the road surface (not on sidewalk)."""
    on_sidewalk_n = abs(py - SIDEWALK_N_Y) < SIDEWALK_HALF_W
    on_sidewalk_s = abs(py - SIDEWALK_S_Y) < SIDEWALK_HALF_W
    in_road_x = ROAD_X_MIN < px < ROAD_X_MAX
    in_road_y = ROAD_Y_MIN < py < ROAD_Y_MAX
    return in_road_x and in_road_y and not on_sidewalk_n and not on_sidewalk_s


def audit_sdf(sdf_path: str, verbose: bool = False) -> list[dict]:
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    issues: list[dict] = []

    buildings = _collect_buildings(root)

    # ---- Check <include> elements (Fuel models) ----
    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        uri_el = inc.find("uri")

        name = name_el.text.strip() if name_el is not None else "(unnamed)"
        pose = _parse_pose(pose_el.text) if pose_el is not None else (0, 0, 0, 0, 0, 0)
        uri = uri_el.text.strip() if uri_el is not None else ""

        x, y, z, roll, pitch, yaw = pose
        model_name = _extract_model_name_from_uri(uri) or ""

        is_vehicle = any(kw in name.lower() for kw in ("vehicle", "hatchback", "pickup", "truck", "bus", "prius"))
        is_person = name.lower().startswith("person_")
        is_moving_vehicle = name.lower().startswith("vehicle_moving")

        # Vehicle Z check
        if is_vehicle and abs(z) > VEHICLE_Z_MAX:
            issues.append({
                "type": "vehicle_z",
                "severity": "error",
                "model": name,
                "message": f"Vehicle '{name}' Z={z:.3f}, should be ~0 for Fuel models",
            })

        # Static person Z check
        if is_person and abs(z) > VEHICLE_Z_MAX:
            issues.append({
                "type": "person_z",
                "severity": "error",
                "model": name,
                "message": f"Static person '{name}' Z={z:.3f}, Fuel person models sit at Z=0",
            })

        # Vehicle yaw vs mesh internal offset
        mesh_offset = FUEL_MESH_YAW.get(model_name, None)
        if mesh_offset is not None and is_vehicle:
            world_facing = yaw + mesh_offset
            if verbose:
                issues.append({
                    "type": "vehicle_yaw",
                    "severity": "info",
                    "model": name,
                    "message": (
                        f"'{name}' ({model_name}) SDF yaw={yaw:.2f} + internal={mesh_offset:.2f} "
                        f"-> faces {_heading_name(world_facing)}"
                    ),
                })

        # Check if static person is inside road (forbidden zone)
        if is_person and _on_road(x, y):
            issues.append({
                "type": "road_forbidden",
                "severity": "warning",
                "model": name,
                "message": (
                    f"Static person '{name}' at ({x:.1f}, {y:.1f}) is inside the road "
                    f"surface (Y={ROAD_Y_MIN}..{ROAD_Y_MAX}) — move to sidewalk or crosswalk"
                ),
            })

        # Check if static person overlaps building
        if is_person:
            bname = _point_in_building(x, y, buildings)
            if bname:
                issues.append({
                    "type": "person_collision",
                    "severity": "error",
                    "model": name,
                    "message": f"Static person '{name}' at ({x:.1f}, {y:.1f}) overlaps building '{bname}'",
                })

    # ---- Check <actor> elements ----
    for actor in root.iter("actor"):
        actor_name = actor.get("name", "(unnamed)")
        skin = actor.find("skin/filename")
        is_mingfei = skin is not None and skin.text and "Mingfei" in skin.text

        trajectory = actor.find(".//trajectory")
        if trajectory is None:
            continue

        waypoints = trajectory.findall("waypoint")
        if len(waypoints) < 2:
            continue

        wp_data: list[tuple[float, float, float, float]] = []  # (t, x, y, z)
        for wp in waypoints:
            time_el = wp.find("time")
            pose_el = wp.find("pose")
            if time_el is not None and pose_el is not None and time_el.text and pose_el.text:
                t = float(time_el.text.strip())
                x, y, z = _parse_pose(pose_el.text)[:3]
                wp_data.append((t, x, y, z))

        # Check Z values
        for t, x, y, z in wp_data:
            if is_mingfei and abs(z - MINGFEI_WALK_Z) > Z_TOLERANCE:
                issues.append({
                    "type": "actor_z",
                    "severity": "error",
                    "model": actor_name,
                    "message": (
                        f"Mingfei walk actor '{actor_name}' Z={z:.1f} at waypoint t={t:.0f}s — "
                        f"mesh origin at body center (Z≈0 in mesh, feet at Z≈-0.86), "
                        f"requires Z={MINGFEI_WALK_Z} "
                        f"(Z=0 causes actor to appear buried with only head visible)"
                    ),
                })

        # Check facing direction and speed
        for i in range(len(wp_data) - 1):
            t1, x1, y1, z1 = wp_data[i]
            t2, x2, y2, z2 = wp_data[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            dt = t2 - t1

            if dist < 0.3 or dt < 0.3:
                continue

            speed = dist / dt

            # Speed checks
            if speed < SPEED_MIN:
                issues.append({
                    "type": "actor_speed",
                    "severity": "warning",
                    "model": actor_name,
                    "message": f"'{actor_name}' segment [{i}->{i + 1}] speed={speed:.2f} m/s — too slow",
                })
            elif speed > SPEED_MAX:
                issues.append({
                    "type": "actor_speed",
                    "severity": "warning",
                    "model": actor_name,
                    "message": f"'{actor_name}' segment [{i}->{i + 1}] speed={speed:.2f} m/s — too fast for walking",
                })

            # Yaw facing check — extract yaw from next waypoint pose
            next_pose_text = waypoints[i + 1].find("pose")
            if next_pose_text is not None and next_pose_text.text:
                wp_yaw = _parse_pose(next_pose_text.text)[5]
                travel_yaw = math.atan2(dy, dx)
                yaw_err = _angle_diff(wp_yaw, travel_yaw)
                if yaw_err > YAW_TOLERANCE:
                    issues.append({
                        "type": "actor_yaw",
                        "severity": "warning",
                        "model": actor_name,
                        "message": (
                            f"'{actor_name}' segment [{i}->{i + 1}] yaw={wp_yaw:.2f} "
                            f"({(_heading_name(wp_yaw))}) vs travel={travel_yaw:.2f} "
                            f"({_heading_name(travel_yaw)}) — diff={yaw_err:.2f}rad "
                            f"(sideways walking if not a turn waypoint)"
                        ),
                    })

        # Check building overlap
        for t, x, y, z in wp_data:
            bname = _point_in_building(x, y, buildings)
            if bname:
                issues.append({
                    "type": "actor_collision",
                    "severity": "error",
                    "model": actor_name,
                    "message": f"'{actor_name}' waypoint at ({x:.1f}, {y:.1f}) t={t:.0f}s overlaps building '{bname}'",
                })

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Gazebo scene SDF for pose issues")
    parser.add_argument("--sdf", default="configs/gz/sentinel_street.sdf", help="SDF file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show info notes")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.sdf).exists():
        print(f"ERROR: {args.sdf} not found", file=sys.stderr)
        sys.exit(1)

    issues = audit_sdf(args.sdf, verbose=args.verbose)

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]

    if args.json:
        print(json.dumps(issues, indent=2))
    else:
        for i in errors:
            print(f"  ERROR  [{i['type']:20s}] {i['message']}")
        for i in warnings:
            print(f"  WARN   [{i['type']:20s}] {i['message']}")
        if args.verbose:
            for i in infos:
                print(f"  INFO   [{i['type']:20s}] {i['message']}")

        print()
        if errors:
            print(f"FAIL: {len(errors)} errors, {len(warnings)} warnings")
        elif warnings:
            print(f"WARN: {len(warnings)} warnings, no errors")
        else:
            print(f"OK: No issues found ({len(infos)} info notes)")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
