"""Audit generated rural ISR SDF worlds for correctness.

Checks: Fuel asset cache, broken URIs, waypoint Z alignment, vehicle slope,
terrain bounds, target counts, concealment distribution, performance estimate.

Usage:
    python -m apps.tools.audit_rural_isr_scene --world 1779343687303 --density medium
    python -m apps.tools.audit_rural_isr_scene --world 1779343687303 --density medium --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from apps.tools.import_fuel_isr_assets import verify_isr_all
from apps.tools.place_real_terrain_vegetation import load_terrain, terrain_z
from apps.tools.place_rural_isr_scene import RURAL_PROFILES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PX4_DIR = PROJECT_ROOT.parent / "PX4-Autopilot"
GZ_WORLDS_DIR = PX4_DIR / "Tools" / "simulation" / "gz" / "worlds"

CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"


@dataclass
class AuditResult:
    level: str  # "ok" | "warn" | "fail"
    category: str
    message: str
    detail: str = ""

    def __str__(self) -> str:
        tag = {"ok": "  OK ", "warn": " WARN", "fail": "FAIL"}[self.level]
        line = f"  [{tag}] {self.category}: {self.message}"
        if self.detail:
            line += f"\n         {self.detail}"
        return line


class AuditReport:
    def __init__(self) -> None:
        self.results: list[AuditResult] = []

    def ok(self, category: str, message: str, detail: str = "") -> None:
        self.results.append(AuditResult("ok", category, message, detail))

    def warn(self, category: str, message: str, detail: str = "") -> None:
        self.results.append(AuditResult("warn", category, message, detail))

    def fail(self, category: str, message: str, detail: str = "") -> None:
        self.results.append(AuditResult("fail", category, message, detail))

    @property
    def n_fails(self) -> int:
        return sum(1 for r in self.results if r.level == "fail")

    @property
    def n_warns(self) -> int:
        return sum(1 for r in self.results if r.level == "warn")

    def print_report(self, verbose: bool = False) -> None:
        for r in self.results:
            if verbose or r.level != "ok":
                print(r)
        print(f"\n  Total: {len(self.results)} checks, "
              f"{self.n_fails} failures, {self.n_warns} warnings")


# ---------------------------------------------------------------------------
# Audit checks
# ---------------------------------------------------------------------------


def check_fuel_assets(report: AuditReport) -> None:
    """Verify all ISR Fuel models are cached."""
    registry = verify_isr_all()
    missing = []
    for key, info in registry.items():
        if not info.cache_dir or not info.cache_dir.exists():
            missing.append(key)

    if missing:
        report.fail("Fuel assets", f"{len(missing)} models not cached",
                    ", ".join(missing))
    else:
        report.ok("Fuel assets", f"All {len(registry)} models cached")


def check_sdf_load(report: AuditReport, sdf_path: Path) -> ET.Element | None:
    """Parse SDF file."""
    if not sdf_path.exists():
        report.fail("SDF file", f"Not found: {sdf_path}")
        return None
    try:
        tree = ET.parse(sdf_path)
        return tree.getroot()
    except ET.ParseError as e:
        report.fail("SDF file", f"XML parse error: {e}")
        return None


def check_model_uris(report: AuditReport, root: ET.Element) -> None:
    """Check all <uri> elements in <include> blocks reference cached models."""
    broken = []
    checked = set()
    for inc in root.iter("include"):
        uri_elem = inc.find("uri")
        if uri_elem is None or uri_elem.text is None:
            continue
        uri = uri_elem.text.strip()
        if uri in checked or uri.startswith("model://"):
            checked.add(uri)
            continue
        if uri.startswith("https://fuel.gazebosim.org"):
            parts = uri.split("/")
            if len(parts) >= 6:
                owner = parts[4]
                model_name = parts[6] if len(parts) > 6 else parts[5]
                cache_dir = CACHE_ROOT / owner.lower() / "models" / model_name.lower()
                if not cache_dir.exists():
                    broken.append(uri)
        checked.add(uri)

    if broken:
        report.warn("Model URIs", f"{len(broken)} URIs not in cache",
                    "\n    ".join(broken[:5]))
    else:
        report.ok("Model URIs", f"All {len(checked)} URIs resolvable")


def _extract_waypoints(root: ET.Element) -> list[dict]:
    """Extract actor waypoint positions from SDF."""
    waypoints = []
    for actor in root.iter("actor"):
        name = actor.get("name", "unknown")
        for wp in actor.iter("waypoint"):
            pose_elem = wp.find("pose")
            if pose_elem is not None and pose_elem.text:
                parts = pose_elem.text.strip().split()
                if len(parts) >= 3:
                    waypoints.append({
                        "actor": name,
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "z": float(parts[2]),
                    })
    return waypoints


def check_waypoint_z_alignment(
    report: AuditReport, root: ET.Element, terrain_name: str,
) -> None:
    """Check actor waypoints are not floating or sunken relative to terrain."""
    try:
        terrain = load_terrain(terrain_name)
    except Exception as e:
        report.warn("Waypoint Z", f"Cannot load terrain: {e}")
        return

    waypoints = _extract_waypoints(root)
    if not waypoints:
        report.ok("Waypoint Z", "No actor waypoints found")
        return

    floating = 0
    sunken = 0
    for wp in waypoints:
        tz = terrain_z(terrain, wp["x"], wp["y"])
        delta = wp["z"] - tz
        if delta > 3.0:
            floating += 1
        elif delta < -0.5:
            sunken += 1

    total = len(waypoints)
    if floating > 0:
        report.warn("Waypoint Z",
                    f"{floating}/{total} waypoints floating (>3m above terrain)")
    if sunken > 0:
        report.warn("Waypoint Z",
                    f"{sunken}/{total} waypoints sunken (>0.5m below terrain)")
    if floating == 0 and sunken == 0:
        report.ok("Waypoint Z", f"All {total} waypoints aligned to terrain")


def check_vehicle_slope(
    report: AuditReport, root: ET.Element, terrain_name: str,
) -> None:
    """Warn if vehicles are placed on steep slopes."""
    try:
        terrain = load_terrain(terrain_name)
    except Exception:
        report.warn("Vehicle slope", "Cannot load terrain for slope check")
        return

    steep = []
    for inc in root.iter("include"):
        name_elem = inc.find("name")
        pose_elem = inc.find("pose")
        if name_elem is None or pose_elem is None:
            continue
        name = name_elem.text or ""
        if not name.startswith("vehicle_"):
            continue
        parts = pose_elem.text.strip().split()
        if len(parts) < 3:
            continue
        x, y = float(parts[0]), float(parts[1])
        radius = 2.0
        z_c = terrain_z(terrain, x, y)
        z_xp = terrain_z(terrain, x + radius, y)
        z_xn = terrain_z(terrain, x - radius, y)
        z_yp = terrain_z(terrain, x, y + radius)
        z_yn = terrain_z(terrain, x, y - radius)
        dx = (z_xp - z_xn) / (2 * radius)
        dy = (z_yp - z_yn) / (2 * radius)
        slope_deg = math.degrees(math.atan(math.sqrt(dx * dx + dy * dy)))
        if slope_deg > 15.0:
            steep.append(f"{name}: {slope_deg:.1f}°")

    if steep:
        report.warn("Vehicle slope", f"{len(steep)} vehicles on steep ground (>15°)",
                    "\n    ".join(steep[:5]))
    else:
        report.ok("Vehicle slope", "All vehicles on flat ground (<15°)")


def check_terrain_bounds(
    report: AuditReport, root: ET.Element, terrain_name: str,
) -> None:
    """Verify all entities are within terrain bounds."""
    try:
        terrain = load_terrain(terrain_name)
    except Exception:
        report.warn("Terrain bounds", "Cannot load terrain")
        return

    if not terrain.pixels:
        report.ok("Terrain bounds", "No terrain pixels — skip bounds check")
        return

    margin = 20.0
    x_min = terrain.offset_x - terrain.width_m / 2 + margin
    x_max = terrain.offset_x + terrain.width_m / 2 - margin
    y_min = terrain.offset_y - terrain.height_m / 2 + margin
    y_max = terrain.offset_y + terrain.height_m / 2 - margin

    oob = []
    for model in root.iter("model"):
        pose_elem = model.find("pose")
        if pose_elem is None or pose_elem.text is None:
            continue
        parts = pose_elem.text.strip().split()
        if len(parts) < 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        name = model.get("name", "?")
        if name.startswith("concealment_zone"):
            continue
        if x < x_min or x > x_max or y < y_min or y > y_max:
            oob.append(f"{name} at ({x:.1f}, {y:.1f})")

    for inc in root.iter("include"):
        pose_elem = inc.find("pose")
        if pose_elem is None or pose_elem.text is None:
            continue
        parts = pose_elem.text.strip().split()
        if len(parts) < 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        if x < x_min or x > x_max or y < y_min or y > y_max:
            name = inc.find("name")
            nm = name.text if name is not None else "?"
            oob.append(f"{nm} at ({x:.1f}, {y:.1f})")

    # Check actor waypoints
    for actor in root.iter("actor"):
        for wp in actor.iter("waypoint"):
            pose = wp.find("pose")
            if pose is None or pose.text is None:
                continue
            parts = pose.text.strip().split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            if x < x_min or x > x_max or y < y_min or y > y_max:
                oob.append(f"actor {actor.get('name', '?')} wp at ({x:.1f}, {y:.1f})")

    if oob:
        report.warn("Terrain bounds", f"{len(oob)} entities outside terrain",
                    "\n    ".join(oob[:5]))
    else:
        report.ok("Terrain bounds", "All entities within terrain")


def check_target_counts(
    report: AuditReport, sdf_path: Path, density: str,
) -> None:
    """Compare actual target counts against density profile expectations."""
    content = sdf_path.read_text()
    profile = RURAL_PROFILES[density]

    # Count via TARGET_METADATA comments
    targets = re.findall(r'TARGET_METADATA:\s*(\{[^}]+\})', content)
    target_types: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    for t_json in targets:
        try:
            meta = json.loads(t_json)
            tt = meta.get("target_type", "unknown")
            target_types[tt] = target_types.get(tt, 0) + 1
            diff = meta.get("expected_difficulty", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1
        except json.JSONDecodeError:
            pass

    n_actors = content.count("<actor")
    n_vehicles = target_types.get("civilian_vehicle", 0) + target_types.get("concealed_vehicle", 0)

    expected_civ = profile["civilian_people"]
    expected_mil = profile["military_people"]
    actual_civ = target_types.get("civilian", 0)
    actual_mil = target_types.get("military_like", 0)

    if actual_civ < expected_civ * 0.8:
        report.warn("Target counts",
                    f"Civilian people: {actual_civ} (expected ~{expected_civ})")
    else:
        report.ok("Target counts",
                  f"Civilian people: {actual_civ} (expected ~{expected_civ})")

    if actual_mil < expected_mil * 0.8:
        report.warn("Target counts",
                    f"Military people: {actual_mil} (expected ~{expected_mil})")
    else:
        report.ok("Target counts",
                  f"Military people: {actual_mil} (expected ~{expected_mil})")

    report.ok("Target metadata",
              f"{len(targets)} targets with metadata, "
              f"difficulties: {difficulties}")


def check_concealment_distribution(report: AuditReport, sdf_path: Path) -> None:
    """Check concealment level distribution from metadata comments."""
    content = sdf_path.read_text()
    targets = re.findall(r'TARGET_METADATA:\s*(\{[^}]+\})', content)

    levels = []
    for t_json in targets:
        try:
            meta = json.loads(t_json)
            cl = meta.get("concealment_level", 0.0)
            levels.append(float(cl))
        except (json.JSONDecodeError, ValueError):
            pass

    if not levels:
        report.ok("Concealment", "No concealment data found")
        return

    open_count = sum(1 for l in levels if l < 0.3)
    partial_count = sum(1 for l in levels if 0.3 <= l < 0.7)
    high_count = sum(1 for l in levels if l >= 0.7)
    avg = sum(levels) / len(levels)

    report.ok("Concealment",
              f"open={open_count}, partial={partial_count}, high={high_count} "
              f"(avg concealment: {avg:.2f})")

    military_levels = []
    for t_json in targets:
        try:
            meta = json.loads(t_json)
            if meta.get("target_type") == "military_like":
                military_levels.append(float(meta.get("concealment_level", 0.0)))
        except (json.JSONDecodeError, ValueError):
            pass

    if military_levels:
        mil_avg = sum(military_levels) / len(military_levels)
        if mil_avg < 0.3:
            report.warn("Concealment",
                        f"Military targets avg concealment only {mil_avg:.2f} "
                        f"(expected > 0.3)")
        else:
            report.ok("Concealment",
                      f"Military targets avg concealment: {mil_avg:.2f}")


def check_performance_estimate(report: AuditReport, sdf_path: Path) -> None:
    """Estimate rendering performance based on entity count."""
    content = sdf_path.read_text()

    n_models = content.count("<model")
    n_actors = content.count("<actor")
    n_includes = content.count("<include")
    total = n_models + n_actors + n_includes

    if total > 500:
        report.warn("Performance",
                    f"Very high entity count: {total} (expect <15 FPS on RTX 4090)")
    elif total > 300:
        report.warn("Performance",
                    f"High entity count: {total} (expect ~20-30 FPS)")
    elif total > 150:
        report.ok("Performance",
                  f"Moderate entity count: {total} (expect ~30-45 FPS)")
    else:
        report.ok("Performance",
                  f"Low entity count: {total} (expect ~50+ FPS)")


def check_reid_markers(report: AuditReport, sdf_path: Path) -> None:
    """Verify re-ID markers exist for targets."""
    content = sdf_path.read_text()
    n_markers = content.count('name="marker_')
    targets = re.findall(r'TARGET_METADATA:', content)
    n_targets = len(targets)

    if n_targets == 0:
        report.warn("Re-ID markers", "No targets with metadata found")
        return

    # Allow some tolerance — static persons and vehicles should have markers
    ratio = n_markers / n_targets
    if ratio < 0.5:
        report.warn("Re-ID markers",
                    f"Only {n_markers} markers for {n_targets} targets ({ratio:.0%})")
    else:
        report.ok("Re-ID markers",
                  f"{n_markers} markers for {n_targets} targets")


def check_entities_z_alignment(
    report: AuditReport, root: ET.Element, terrain_name: str,
) -> None:
    """Check that all static entities (standing people, vehicles, vegetation, sheds) are snapped to ground height."""
    try:
        terrain = load_terrain(terrain_name)
    except Exception as e:
        report.warn("Entities Z", f"Cannot load terrain: {e}")
        return

    oob_float = []
    oob_sink = []
    
    # Audit includes (vehicles, standing people)
    for inc in root.iter("include"):
        name_elem = inc.find("name")
        pose_elem = inc.find("pose")
        uri_elem = inc.find("uri")
        if name_elem is None or pose_elem is None:
            continue
        name = name_elem.text or ""
        uri = uri_elem.text if uri_elem is not None else ""
        
        z_off = 0.0
        if "Walking person" in uri or "walking_person" in name:
            z_off = 0.02
            
        parts = pose_elem.text.strip().split()
        if len(parts) < 3:
            continue
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        
        # Skip markers or helipad
        if name.startswith("marker_") or name == "helipad" or "helipad" in uri:
            continue
            
        expected_z = terrain_z(terrain, x, y) + z_off
        diff = z - expected_z
        
        if diff > 0.15:
            oob_float.append(f"{name} at ({x:.1f}, {y:.1f}) floating by {diff:.2f}m (Z={z:.2f}, expected={expected_z:.2f})")
        elif diff < -0.10:
            oob_sink.append(f"{name} at ({x:.1f}, {y:.1f}) sunken by {abs(diff):.2f}m (Z={z:.2f}, expected={expected_z:.2f})")
            
    # Audit vegetation/shed models (which are inline models, not includes)
    for model in root.iter("model"):
        name = model.get("name", "")
        if name.startswith("veg_") or name.startswith("tl_") or name.startswith("shed_") or name.startswith("road_"):
            pose_elem = model.find("pose")
            if pose_elem is None or pose_elem.text is None:
                continue
            parts = pose_elem.text.strip().split()
            if len(parts) < 3:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            
            z_off = 0.02 if name.startswith("road_") else 0.0
            expected_z = terrain_z(terrain, x, y) + z_off
            diff = z - expected_z
            
            # Allow slightly larger tolerance for vegetation
            if name.startswith("road_") or name.startswith("shed_"):
                max_float, max_sink = 0.15, 0.10
            else:
                max_float, max_sink = 0.25, 0.15
                
            if diff > max_float:
                oob_float.append(f"{name} at ({x:.1f}, {y:.1f}) floating by {diff:.2f}m (Z={z:.2f}, expected={expected_z:.2f})")
            elif diff < -max_sink:
                oob_sink.append(f"{name} at ({x:.1f}, {y:.1f}) sunken by {abs(diff):.2f}m (Z={z:.2f}, expected={expected_z:.2f})")

    total_oob = len(oob_float) + len(oob_sink)
    if total_oob > 0:
        report.warn("Entities Z", f"{total_oob} static entities are misaligned with terrain",
                    "\n    ".join((oob_float + oob_sink)[:8]))
    else:
        report.ok("Entities Z", "All static entities are perfectly snapped to terrain ground level (floating <= 0.15m, sunken <= 0.10m)")


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def audit_world(
    terrain_name: str,
    density: str,
    verbose: bool = False,
    output_dir: Path | None = None,
) -> AuditReport:
    """Run full audit on a generated rural ISR world."""
    report = AuditReport()
    world_name = f"{terrain_name}_isr_rural_{density}"
    base_dir = Path(output_dir) if output_dir else GZ_WORLDS_DIR
    sdf_path = base_dir / f"{world_name}.sdf"

    print(f"Auditing: {sdf_path.name}")
    print(f"Density:  {density}")
    print()

    # 1. Fuel assets
    check_fuel_assets(report)

    # 2. SDF file load
    root = check_sdf_load(report, sdf_path)

    if root is not None:
        # 3. Model URIs
        check_model_uris(report, root)

        # 4. Waypoint Z alignment
        check_waypoint_z_alignment(report, root, terrain_name)

        # 4.1 Entities Z alignment (floating/sunken)
        check_entities_z_alignment(report, root, terrain_name)

        # 5. Vehicle slope
        check_vehicle_slope(report, root, terrain_name)

        # 6. Terrain bounds
        check_terrain_bounds(report, root, terrain_name)

    # 7. Target counts (can work from file alone)
    if sdf_path.exists():
        check_target_counts(report, sdf_path, density)

        # 8. Concealment distribution
        check_concealment_distribution(report, sdf_path)

        # 9. Performance estimate
        check_performance_estimate(report, sdf_path)

        # 10. Re-ID markers
        check_reid_markers(report, sdf_path)

    report.print_report(verbose)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit generated rural ISR SDF world",
    )
    parser.add_argument("--world", required=True, help="World/model name")
    parser.add_argument("--density", choices=["light", "medium", "heavy", "camera_lite", "realistic_lite"],
                        default="medium")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Override SDF directory")
    args = parser.parse_args()

    report = audit_world(args.world, args.density, args.verbose, args.output_dir)

    if report.n_fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
