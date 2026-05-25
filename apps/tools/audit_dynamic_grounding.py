"""Audit dynamic world grounding using calibrated visual contact points.

Usage:
    python3 -m apps.tools.audit_dynamic_grounding --world 1779343687303_isr_rural_dynamic_v3
    python3 -m apps.tools.audit_dynamic_grounding --sdf /path/to/world.sdf --terrain 1779343687303
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from apps.tools.place_real_terrain_vegetation import load_terrain
from apps.tools.generate_v3_world import (
    PX4_WORLDS_DIR,
    TARGET_WORLD,
    AssetCalibration,
    classify_entity,
    sample_terrain_z,
)


def _parse_pose(pose_text: str | None) -> tuple[float, float, float, float, float, float]:
    if not pose_text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = [float(v) for v in pose_text.strip().split()[:6]]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals)  # type: ignore[return-value]


def _record(
    *,
    name: str,
    asset_label: str,
    calibration: AssetCalibration,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    terrain_z: float,
    note: str = "",
) -> dict:
    origin_clearance = z - terrain_z
    visual_contact_z = z - calibration.z_offset_m
    clearance = visual_contact_z - terrain_z
    floating_fail = clearance > calibration.floating_tol_m
    penetration_fail = clearance < -calibration.penetration_tol_m
    status = "FAIL" if floating_fail or penetration_fail else "PASS"
    reason = "OK"
    if floating_fail:
        reason = f"floating {clearance:+.3f}m > +{calibration.floating_tol_m:.2f}m"
    elif penetration_fail:
        reason = f"sunk {clearance:+.3f}m < -{calibration.penetration_tol_m:.2f}m"
    elif calibration.approximate:
        reason = "approximate calibration"
    return {
        "name": name,
        "asset_type": calibration.asset_type,
        "asset_label": asset_label,
        "x": x,
        "y": y,
        "terrain_z": terrain_z,
        "origin_z": z,
        "z_offset": calibration.z_offset_m,
        "final_pose_z": z,
        "visual_contact_z": visual_contact_z,
        "origin_clearance": origin_clearance,
        "clearance": clearance,
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw),
        "status": status,
        "reason": reason,
        "contact_type": calibration.contact_type,
        "notes": note or calibration.notes,
        "approximate": calibration.approximate,
    }


def _iter_records(root: ET.Element, terrain) -> list[dict]:
    records: list[dict] = []
    world = root.find("world")
    if world is None:
        raise RuntimeError("Missing <world> element")

    for child in list(world):
        tag = child.tag.split("}")[-1]
        if tag == "include":
            name = (child.findtext("name") or "").strip()
            if "x500" in name.lower() or "mono_cam" in name.lower():
                continue
            uri = (child.findtext("uri") or "").strip()
            calibration = classify_entity(name, "include", uri)
            if calibration is None:
                continue
            x, y, z, roll, pitch, yaw = _parse_pose(child.findtext("pose"))
            terrain_z = sample_terrain_z(terrain, x, y)
            records.append(
                _record(
                    name=name,
                    asset_label=uri,
                    calibration=calibration,
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    terrain_z=terrain_z,
                )
            )
        elif tag == "model":
            name = child.get("name", "")
            if "x500" in name.lower() or "mono_cam" in name.lower() or name.startswith("marker_"):
                continue
            calibration = classify_entity(name, "model")
            if calibration is None:
                continue
            x, y, z, roll, pitch, yaw = _parse_pose(child.findtext("pose"))
            terrain_z = sample_terrain_z(terrain, x, y)
            records.append(
                _record(
                    name=name,
                    asset_label=name,
                    calibration=calibration,
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    terrain_z=terrain_z,
                )
            )
        elif tag == "actor":
            name = child.get("name", "")
            calibration = classify_entity(name, "actor")
            if calibration is None:
                continue
            # Actor base pose
            x, y, z, roll, pitch, yaw = _parse_pose(child.findtext("pose"))
            terrain_z = sample_terrain_z(terrain, x, y)
            records.append(
                _record(
                    name=name,
                    asset_label=name,
                    calibration=calibration,
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    terrain_z=terrain_z,
                    note="actor root pose",
                )
            )
            # Waypoints
            for idx, wp in enumerate(child.findall(".//trajectory/waypoint")):
                x, y, z, roll, pitch, yaw = _parse_pose(wp.findtext("pose"))
                terrain_z = sample_terrain_z(terrain, x, y)
                records.append(
                    _record(
                        name=f"{name}_wp{idx:03d}",
                        asset_label=name,
                        calibration=calibration,
                        x=x,
                        y=y,
                        z=z,
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        terrain_z=terrain_z,
                        note="actor trajectory waypoint",
                    )
                )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dynamic_v3 grounding using calibrated contact points")
    parser.add_argument("--world", help="World SDF basename, e.g. 1779343687303_isr_rural_dynamic_v3")
    parser.add_argument("--sdf", help="Explicit path to SDF world file")
    parser.add_argument("--terrain", default="1779343687303", help="Terrain model name for heightmap sampling")
    parser.add_argument("--report-json", default="/tmp/isr_dynamic_v3_grounding_audit.json")
    args = parser.parse_args()

    if args.sdf:
        sdf_path = Path(args.sdf)
    else:
        world_name = args.world or TARGET_WORLD
        sdf_path = PX4_WORLDS_DIR / f"{world_name}.sdf"

    if not sdf_path.exists():
        print(f"ERROR: SDF not found: {sdf_path}", file=sys.stderr)
        sys.exit(1)

    terrain = load_terrain(args.terrain)
    root = ET.parse(sdf_path).getroot()
    records = _iter_records(root, terrain)
    failures = [r for r in records if r["status"] == "FAIL"]
    warnings = [r for r in records if r["status"] == "PASS" and r["approximate"]]

    summary = {
        "sdf": str(sdf_path),
        "terrain": args.terrain,
        "audited_records": len(records),
        "failures": len(failures),
        "warnings": len(warnings),
        "records": records,
    }
    Path(args.report_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 140)
    print("CALIBRATED DYNAMIC GROUNDING AUDIT")
    print("=" * 140)
    print(
        f"| {'Entity':<28} | {'Type':<10} | {'X':>7} | {'Y':>7} | {'Terrain':>7} | {'PoseZ':>7} | {'Zoff':>6} | "
        f"{'VisZ':>7} | {'Clr':>7} | {'RPY deg':<21} | {'Status':<6} | Notes |"
    )
    print(
        f"| {'-'*28} | {'-'*10} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | "
        f"{'-'*7} | {'-'*7} | {'-'*21} | {'-'*6} | {'-'*20} |"
    )
    for rec in sorted(records, key=lambda r: (r["status"] != "FAIL", r["asset_type"], r["name"])):
        rpy = f"{rec['roll_deg']:.0f}/{rec['pitch_deg']:.0f}/{rec['yaw_deg']:.0f}"
        print(
            f"| {rec['name']:<28} | {rec['asset_type']:<10} | {rec['x']:>7.2f} | {rec['y']:>7.2f} | "
            f"{rec['terrain_z']:>7.2f} | {rec['final_pose_z']:>7.2f} | {rec['z_offset']:>6.2f} | "
            f"{rec['visual_contact_z']:>7.2f} | {rec['clearance']:>+7.3f} | {rpy:<21} | {rec['status']:<6} | {rec['reason']} |"
        )

    print("=" * 140)
    print(f"Audited records: {len(records)}")
    print(f"Failures:        {len(failures)}")
    print(f"Warnings:        {len(warnings)}")
    print(f"JSON report:     {args.report_json}")
    print("=" * 140)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
