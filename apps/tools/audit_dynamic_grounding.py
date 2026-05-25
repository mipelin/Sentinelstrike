"""Audit terrain grounding of all assets in an SDF world file with strict tolerances.

Part A requirement:
1. Samples terrain height at each asset's x/y.
2. Applies mesh-specific visual-base offsets.
3. Computes final ground clearances.
4. Generates an audit report showing: entity name, x/y/z, terrain height, model offset, clearance, pass/fail.
5. Strict tolerances:
   - people floating <= 0.10 m
   - people penetration <= 0.05 m
   - vehicles floating <= 0.15 m
   - vehicles penetration <= 0.10 m
   - vegetation floating <= 0.15 m
   - buildings/sheds penetration <= 0.10 m

Usage:
    python3 -m apps.tools.audit_dynamic_grounding --sdf PX4-Autopilot/Tools/simulation/gz/worlds/1779343687303_isr_rural_dynamic_v3.sdf --world 1779343687303
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Setup paths relative to script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from apps.tools.place_real_terrain_vegetation import load_terrain, terrain_z


def _parse_pose(pose_text: str) -> tuple[float, float, float, float, float, float]:
    if not pose_text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = pose_text.strip().split()
    vals = [float(v) for v in parts[:6]]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals)  # type: ignore[return-value]


def get_model_z_offset(name: str, uri: str) -> float:
    """Returns the visual base Z-offset relative to model origin."""
    name_l = name.lower()
    uri_l = uri.lower()
    
    # Check if it's an actor mesh or standing person
    if "femalevisitor" in uri_l or "femalevisitor" in name_l:
        return 0.0
    if "walking person" in uri_l or "walking_person" in name_l:
        return 0.02
    if "rescue randy" in uri_l or "rescue_randy" in name_l:
        return 0.0
    if "malevisitoronphone" in uri_l or "malevisitoronphone" in name_l:
        return 0.0
    
    # Check vehicles
    if "pickup" in uri_l or "pickup" in name_l:
        return 0.0
    if "truckbox" in uri_l or "truckbox" in name_l:
        return 0.0
    if "hatchback" in uri_l or "hatchback" in name_l:
        return 0.0
    if "bus" in uri_l or "bus" in name_l:
        return 0.0
        
    # Default visual base offset
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Terrain Grounding Auditor")
    parser.add_argument("--sdf", required=True, help="Path to SDF world file")
    parser.add_argument("--world", required=True, help="Terrain world/model name (e.g. 1779343687303)")
    parser.add_argument("--report-json", help="Optional path to write the full audit as JSON")
    args = parser.parse_args()

    sdf_path = Path(args.sdf)
    if not sdf_path.exists():
        print(f"ERROR: SDF file not found at {sdf_path}")
        sys.exit(1)

    print(f"Loading terrain heightmap for world: {args.world}...")
    terrain = load_terrain(args.world)
    if not terrain.pixels:
        print("ERROR: Heightmap pixels not loaded correctly. Cannot sample heights!")
        sys.exit(1)

    print(f"Parsing SDF: {sdf_path}...")
    tree = ET.parse(str(sdf_path))
    root = tree.getroot()

    audit_records = []
    failures = 0

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        # 1. Check <include> elements (standing people, vehicles, sheds, vegetation)
        if tag == "include":
            name_el = elem.find("name")
            name = name_el.text if name_el is not None and name_el.text else "unknown"
            
            # Skip drone x500 models
            if "x500" in name.lower() or "mono_cam" in name.lower():
                continue
                
            uri_el = elem.find("uri")
            uri = uri_el.text if uri_el is not None and uri_el.text else ""
            
            pose_el = elem.find("pose")
            if pose_el is not None and pose_el.text:
                x, y, z, _, _, _ = _parse_pose(pose_el.text)
                t_z = terrain_z(terrain, x, y)
                z_offset = get_model_z_offset(name, uri)
                
                # Visual base expected altitude
                expected_z = t_z + z_offset
                clearance = z - expected_z  # positive means floating, negative means sunk
                
                # Determine asset type and enforce strict tolerances
                asset_class = "vegetation"
                if "person" in name.lower() or "visitor" in name.lower() or "randy" in name.lower() or "civilian" in name.lower() or "military" in name.lower():
                    asset_class = "person"
                elif "vehicle" in name.lower() or "pickup" in name.lower() or "truck" in name.lower():
                    asset_class = "vehicle"
                elif "shed" in name.lower() or "building" in name.lower() or "house" in name.lower():
                    asset_class = "building"
                
                is_ok = True
                reason = "OK"
                
                if asset_class == "person":
                    # people floating <= 0.10m, penetration <= 0.05m
                    if clearance > 0.10:
                        is_ok = False
                        reason = f"Floating too high ({clearance:.3f}m > 0.10m)"
                    elif clearance < -0.05:
                        is_ok = False
                        reason = f"Penetrated too deep ({clearance:.3f}m < -0.05m)"
                elif asset_class == "vehicle":
                    # vehicles floating <= 0.15m, penetration <= 0.10m
                    if clearance > 0.15:
                        is_ok = False
                        reason = f"Floating too high ({clearance:.3f}m > 0.15m)"
                    elif clearance < -0.10:
                        is_ok = False
                        reason = f"Penetrated too deep ({clearance:.3f}m < -0.10m)"
                elif asset_class == "building":
                    # buildings/sheds penetration <= 0.10m, floating <= 0.05m
                    if clearance > 0.05:
                        is_ok = False
                        reason = f"Floating too high ({clearance:.3f}m > 0.05m)"
                    elif clearance < -0.10:
                        is_ok = False
                        reason = f"Penetrated too deep ({clearance:.3f}m < -0.10m)"
                elif asset_class == "vegetation":
                    # vegetation floating <= 0.15m, penetration can be deep (roots)
                    if clearance > 0.15:
                        is_ok = False
                        reason = f"Floating too high ({clearance:.3f}m > 0.15m)"
                    elif clearance < -0.50:  # Allow deep root penetration, but warn if absurd
                        is_ok = False
                        reason = f"Absurd penetration ({clearance:.3f}m < -0.50m)"
                
                if not is_ok:
                    failures += 1
                
                audit_records.append({
                    "name": name,
                    "class": asset_class,
                    "x": x, "y": y, "z": z,
                    "terrain_height": t_z,
                    "z_offset": z_offset,
                    "clearance": clearance,
                    "status": "PASS" if is_ok else "FAIL",
                    "reason": reason
                })

        # 2. Check <actor> elements (walking trajectories)
        elif tag == "actor":
            actor_name = elem.get("name", "unknown")
            traj = elem.find(".//trajectory")
            if traj is not None:
                wp_idx = 0
                for wp in traj.findall("waypoint"):
                    pose_el = wp.find("pose")
                    if pose_el is not None and pose_el.text:
                        x, y, z, _, _, _ = _parse_pose(pose_el.text)
                        t_z = terrain_z(terrain, x, y)
                        
                        # Determine if actor is vehicle or person
                        is_vehicle_actor = any(
                            k in actor_name.lower()
                            for k in ["patrol", "pickup", "vehicle", "truck", "bus", "car"]
                        )
                        
                        if is_vehicle_actor:
                            z_offset = 0.0
                            clearance = z - t_z
                            asset_class = "vehicle"
                        else:
                            z_offset = 1.0
                            clearance = (z - 1.0) - t_z
                            asset_class = "person"
                        
                        is_ok = True
                        reason = "OK"
                        
                        if asset_class == "person":
                            # Apply people tolerances (float <= 0.10m, penetration <= 0.05m)
                            if clearance > 0.10:
                                is_ok = False
                                reason = f"Waypoint {wp_idx} floating too high ({clearance:.3f}m > 0.10m)"
                            elif clearance < -0.05:
                                is_ok = False
                                reason = f"Waypoint {wp_idx} penetrated too deep ({clearance:.3f}m < -0.05m)"
                        elif asset_class == "vehicle":
                            # Apply vehicle tolerances (float <= 0.15m, penetration <= 0.10m)
                            if clearance > 0.15:
                                is_ok = False
                                reason = f"Waypoint {wp_idx} floating too high ({clearance:.3f}m > 0.15m)"
                            elif clearance < -0.10:
                                is_ok = False
                                reason = f"Waypoint {wp_idx} penetrated too deep ({clearance:.3f}m < -0.10m)"
                            
                        if not is_ok:
                            failures += 1
                            
                        audit_records.append({
                            "name": f"{actor_name}_wp{wp_idx}",
                            "class": f"actor_{asset_class}",
                            "x": x, "y": y, "z": z,
                            "terrain_height": t_z,
                            "z_offset": z_offset,
                            "clearance": clearance,
                            "status": "PASS" if is_ok else "FAIL",
                            "reason": reason
                        })
                        wp_idx += 1

    # 3. Check regular <model> elements that are not includes (like roads, sheds placed directly)
    # Wait, some sheds are models. Let's make sure we check them if they are static visual-only buildings
    for elem in root.findall(".//model"):
        name = elem.get("name", "")
        # If it has a static tag and is not a sub-link
        static_el = elem.find("static")
        if static_el is not None and static_el.text == "true":
            # Skip includes that are parsed as sub-models by ET
            if any(rec["name"] == name for rec in audit_records):
                continue
            if "x500" in name.lower() or "mono_cam" in name.lower() or "road" in name.lower() or "marker" in name.lower():
                continue
                
            pose_el = elem.find("pose")
            if pose_el is not None and pose_el.text:
                x, y, z, _, _, _ = _parse_pose(pose_el.text)
                t_z = terrain_z(terrain, x, y)
                
                # Check if it's a building/shed or vegetation model
                asset_class = "building" if "shed" in name.lower() or "building" in name.lower() else "vegetation"
                z_offset = 0.0
                expected_z = t_z + z_offset
                clearance = z - expected_z
                
                is_ok = True
                reason = "OK"
                
                if asset_class == "building":
                    if clearance > 0.05:
                        is_ok = False
                        reason = f"Floating too high ({clearance:.3f}m > 0.05m)"
                    elif clearance < -0.10:
                        is_ok = False
                        reason = f"Penetrated too deep ({clearance:.3f}m < -0.10m)"
                else:
                    if clearance > 0.15:
                        is_ok = False
                        reason = f"Floating too high ({clearance:.3f}m > 0.15m)"
                    elif clearance < -0.50:
                        is_ok = False
                        reason = f"Absurd penetration ({clearance:.3f}m < -0.50m)"
                        
                if not is_ok:
                    failures += 1
                    
                audit_records.append({
                    "name": name,
                    "class": asset_class,
                    "x": x, "y": y, "z": z,
                    "terrain_height": t_z,
                    "z_offset": z_offset,
                    "clearance": clearance,
                    "status": "PASS" if is_ok else "FAIL",
                    "reason": reason
                })

    summary = {
        "sdf": str(sdf_path),
        "world": args.world,
        "total_audited_assets": len(audit_records),
        "total_failures": failures,
        "records": audit_records,
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Output beautiful Markdown Report
    print("\n" + "=" * 80)
    print("TERRAIN GROUNDING AUDIT REPORT")
    print("=" * 80)
    print(f"| {'Entity Name':<35} | {'Class':<12} | {'X/Y/Z Placement':<24} | {'Terrain Ht':<10} | {'Z Offset':<10} | {'Clearance':<10} | {'Status':<6} |")
    print(f"| {'-':-<35} | {'-':-<12} | {'-':-<24} | {'-':-<10} | {'-':-<10} | {'-':-<10} | {'-':-<6} |")
    for r in sorted(audit_records, key=lambda x: (x["status"] == "PASS", x["class"], x["name"])):
        xyz = f"{r['x']:.1f},{r['y']:.1f},{r['z']:.2f}"
        print(f"| {r['name']:<35} | {r['class']:<12} | {xyz:<24} | {r['terrain_height']:<10.2f} | {r['z_offset']:<10.2f} | {r['clearance']:<+10.3f} | {r['status']:<6} |")
        if r["status"] == "FAIL":
            print(f"  --> FAIL REASON: {r['reason']}")
            
    print("=" * 80)
    print(f"Total audited assets: {len(audit_records)}")
    print(f"Total failures:       {failures}")
    if args.report_json:
        print(f"JSON report:          {args.report_json}")
    print("=" * 80)

    if failures > 0:
        print("\nCRITICAL FAILURE: One or more assets violate terrain grounding tolerances.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All assets are perfectly grounded within strict tolerances!")
        sys.exit(0)


if __name__ == "__main__":
    main()
