"""Audit realistic scene SDF for correctness issues.

Checks:
- No floating/sunken entities (terrain Z alignment)
- No sensor leakage from vehicle models
- Model URIs resolve to cached Fuel models
- Vehicle slope warnings
- Actor waypoint terrain tracking
- Performance warnings for high entity count

Usage:
    python -m apps.tools.audit_realistic_scene --world 1779343687303 --density light
    python -m apps.tools.audit_realistic_scene --sdf /path/to/world.sdf --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from apps.tools.place_real_terrain_vegetation import load_terrain, terrain_z

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PX4_DIR = PROJECT_ROOT.parent / "PX4-Autopilot"
GZ_WORLDS_DIR = PX4_DIR / "Tools" / "simulation" / "gz" / "worlds"
CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"

MAX_VEHICLES = 12
MAX_PEOPLE = 30
MAX_TOTAL_ENTITIES = 80


def _find_sdf(world: str, density: str) -> Path | None:
    name = f"{world}_realistic_{density}.sdf"
    for d in [GZ_WORLDS_DIR, PROJECT_ROOT / "configs" / "gz"]:
        p = d / name
        if p.exists():
            return p
    return None


def _check_fuel_uris(root: ET.Element) -> list[dict]:
    issues = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in ("uri", "filename") and elem.text:
            uri = elem.text.strip()
            if uri.startswith("https://fuel.gazebosim.org/"):
                # Extract owner/model from URI
                parts = uri.replace("https://fuel.gazebosim.org/1.0/", "").split("/")
                if len(parts) >= 3:
                    owner = parts[0]
                    model = parts[2]
                    cache_dir = CACHE_ROOT / owner.lower() / "models" / model.lower()
                    if not cache_dir.exists():
                        issues.append({
                            "type": "missing_model",
                            "severity": "error",
                            "model": f"{owner}/{model}",
                            "message": f"Fuel model not cached: {owner}/{model}",
                        })
    return issues


def _parse_pose(pose_text: str) -> tuple[float, float, float, float, float, float]:
    if not pose_text:
        return (0, 0, 0, 0, 0, 0)
    parts = pose_text.strip().split()
    vals = [float(v) for v in parts[:6]]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals)  # type: ignore[return-value]


def _check_terrain_z_alignment(root: ET.Element, terrain_info: object) -> list[dict]:
    from apps.tools.place_real_terrain_vegetation import TerrainInfo

    if not isinstance(terrain_info, TerrainInfo) or not terrain_info.pixels:
        return []

    issues = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        # Check <include> elements (vehicles + standing people)
        if tag == "include":
            name_el = elem.find("name")
            name = name_el.text if name_el is not None and name_el.text else "unknown"
            pose_el = elem.find("pose")
            if pose_el is not None and pose_el.text:
                x, y, z, _, _, _ = _parse_pose(pose_el.text)
                expected = terrain_z(terrain_info, x, y)
                diff = abs(z - expected)
                if diff > 1.0:
                    issues.append({
                        "type": "z_misalignment",
                        "severity": "warning" if diff < 3.0 else "error",
                        "model": name,
                        "message": f"{name} at ({x:.1f},{y:.1f}) Z={z:.2f}, terrain={expected:.2f} (diff={diff:.2f})",
                    })

        # Check <actor> waypoints
        if tag == "actor":
            actor_name = elem.get("name", "unknown")
            traj = elem.find(".//trajectory")
            if traj is not None:
                for wp in traj.findall("waypoint"):
                    pose_el = wp.find("pose")
                    if pose_el is not None and pose_el.text:
                        x, y, z, _, _, _ = _parse_pose(pose_el.text)
                        expected = terrain_z(terrain_info, x, y) + 1.0  # actors stand at terrain + 1.0
                        diff = abs(z - expected)
                        if diff > 2.0:
                            issues.append({
                                "type": "actor_z_drift",
                                "severity": "warning",
                                "model": actor_name,
                                "message": f"{actor_name} waypoint Z={z:.2f}, expected={expected:.2f} (diff={diff:.2f})",
                            })

    return issues


def _check_sensor_leakage(root: ET.Element) -> list[dict]:
    issues = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "sensor":
            parent = elem.getparent()
            if parent is not None:
                parent_tag = parent.tag.split("}")[-1] if "}" in parent.tag else parent.tag
                if parent_tag == "link":
                    model = parent.getparent()
                    model_name = model.get("name", "unknown") if model is not None else "unknown"
                    # Allow sensors only in x500 drone model
                    if "x500" not in model_name.lower() and "mono_cam" not in model_name.lower():
                        stype = elem.get("type", "unknown")
                        issues.append({
                            "type": "sensor_leak",
                            "severity": "warning",
                            "model": model_name,
                            "message": f"Sensor ({stype}) in non-drone model: {model_name}",
                        })
    return issues


def _check_vehicle_slope(root: ET.Element, terrain_info: object) -> list[dict]:
    from apps.tools.place_real_terrain_vegetation import TerrainInfo

    if not isinstance(terrain_info, TerrainInfo) or not terrain_info.pixels:
        return []

    issues = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "include":
            name_el = elem.find("name")
            if name_el is not None and name_el.text and name_el.text.startswith("vehicle_"):
                pose_el = elem.find("pose")
                if pose_el is not None and pose_el.text:
                    x, y, _, _, _, _ = _parse_pose(pose_el.text)
                    from apps.tools.place_realistic_targets import terrain_slope
                    slope = terrain_slope(terrain_info, x, y)
                    if slope > math.radians(15):
                        issues.append({
                            "type": "steep_vehicle",
                            "severity": "warning",
                            "model": name_el.text,
                            "message": f"{name_el.text} on {math.degrees(slope):.1f} degree slope at ({x:.1f},{y:.1f})",
                        })
    return issues


def _check_entity_counts(root: ET.Element) -> list[dict]:
    issues = []
    n_actors = len(root.findall(".//{http://www.w3.org/1999/xhtml}actor")) + \
               len([e for e in root.iter() if e.tag == "actor" or e.tag.endswith("}actor")])
    n_includes = len([e for e in root.iter()
                      if (e.tag == "include" or e.tag.endswith("}include"))])
    n_vehicles = len([e for e in root.iter()
                      if (e.tag == "include" or e.tag.endswith("}include"))
                      and e.find("name") is not None
                      and e.find("name").text is not None
                      and e.find("name").text.startswith("vehicle_")])

    if n_vehicles > MAX_VEHICLES:
        issues.append({
            "type": "perf_warning",
            "severity": "warning",
            "model": "",
            "message": f"High vehicle count ({n_vehicles} > {MAX_VEHICLES}) may affect RTF",
        })

    n_people_actors = n_actors
    if n_people_actors > MAX_PEOPLE:
        issues.append({
            "type": "perf_warning",
            "severity": "warning",
            "model": "",
            "message": f"High people count ({n_people_actors} > {MAX_PEOPLE}) may affect RTF",
        })

    return issues


def audit_realistic_sdf(
    sdf_path: Path | str,
    terrain_name: str,
    verbose: bool = False,
) -> list[dict]:
    tree = ET.parse(str(sdf_path))
    root = tree.getroot()
    terrain = load_terrain(terrain_name)

    all_issues: list[dict] = []
    all_issues.extend(_check_fuel_uris(root))
    all_issues.extend(_check_terrain_z_alignment(root, terrain))
    all_issues.extend(_check_sensor_leakage(root))
    all_issues.extend(_check_vehicle_slope(root, terrain))
    all_issues.extend(_check_entity_counts(root))

    return all_issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit realistic scene SDF")
    parser.add_argument("--world", help="Terrain world name (e.g. 1779343687303)")
    parser.add_argument("--density", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--sdf", help="Direct path to SDF file")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.sdf:
        sdf_path = Path(args.sdf)
        terrain_name = args.world or "unknown"
    elif args.world:
        sdf_path = _find_sdf(args.world, args.density)
        terrain_name = args.world
        if sdf_path is None:
            print(f"ERROR: SDF not found for {args.world}_realistic_{args.density}")
            sys.exit(1)
    else:
        parser.error("Specify --world or --sdf")
        return

    print(f"Auditing: {sdf_path}")
    issues = audit_realistic_sdf(sdf_path, terrain_name, verbose=args.verbose)

    if args.json:
        print(json.dumps(issues, indent=2))
        return

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    for i in issues:
        prefix = "ERROR" if i["severity"] == "error" else "WARN"
        print(f"  [{prefix}] {i['message']}")

    print()
    print(f"Total: {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        sys.exit(1)
    if not issues:
        print("No issues found.")


if __name__ == "__main__":
    main()
