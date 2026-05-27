"""Generate temporary simplified A/B validation worlds from dynamic_v3.

These scenarios keep the same terrain/camera baseline while reducing visual
ambiguity so we can separate tracker/scenario problems from controller issues.
"""

from __future__ import annotations

import copy
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PX4_WORLDS_DIR = PROJECT_ROOT.parent / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "worlds"
sys.path.append(str(PROJECT_ROOT))

from apps.tools.generate_v3_world import calibration_for_key, sample_terrain_z
from apps.tools.place_real_terrain_vegetation import load_terrain

SOURCE_WORLD = "1779343687303_isr_rural_dynamic_v3.sdf"
TERRAIN = load_terrain("1779343687303")

CUSTOM_PERSON_ROUTE = [
    (224.0, -350.0),
    (230.0, -350.0),
]
CUSTOM_VEHICLE_ROUTE = [
    (188.0, -330.0),
    (208.0, -320.0),
    (228.0, -330.0),
    (208.0, -340.0),
    (188.0, -330.0),
]

SCENARIOS = {
    "1779343687303_isr_rural_simple_single_person": {
        "actors": set(),
        "includes": {"person_walking_test_001"},
        "models": {"road_0001", "road_0002", "road_0003", "road_0004"},
    },
    "1779343687303_isr_rural_simple_single_vehicle": {
        "actors": set(),
        "includes": {"vehicle_dynamic_pickup_002"},
        "models": {"road_0001", "road_0002", "road_0003", "road_0004"},
    },
    "1779343687303_isr_rural_simple_person_vehicle_separated": {
        "actors": {"actor_crossing_main"},
        "includes": {"vehicle_dynamic_hatchback_001"},
        "models": {"road_0001", "road_0002", "road_0003", "road_0004"},
    },
}


def _include_name(elem: ET.Element) -> str | None:
    return elem.findtext("name")


def _keep_element(elem: ET.Element, spec: dict) -> bool:
    if elem.tag in {"gravity", "spherical_coordinates", "physics", "light", "plugin"}:
        return True
    if elem.tag == "include":
        name = _include_name(elem)
        uri = elem.findtext("uri", "")
        # Drop terrain model (not needed for simplified validation worlds)
        if "1779343687303" in uri:
            return False
        # Keep unnamed includes (helipad, etc.) and named includes in spec
        if not name:
            return True
        return name in spec["includes"]
    if elem.tag == "actor":
        return elem.get("name") in spec["actors"]
    if elem.tag == "model":
        return elem.get("name") in spec["models"]
    return False


def _format_pose(x: float, y: float, z: float, yaw: float) -> str:
    return f"{x:.2f} {y:.2f} {z:.2f} 0.0000 0.0000 {yaw:.4f}"


def _rewrite_actor_route(
    actor: ET.Element,
    points_xy: list[tuple[float, float]],
    speed_mps: float = 1.2,
    extra_z_m: float = 0.0,
) -> None:
    calibration = calibration_for_key("mingfei_walk")
    yaw = 0.0
    if len(points_xy) > 1:
        x0, y0 = points_xy[0]
        x1, y1 = points_xy[1]
        yaw = math.atan2(y1 - y0, x1 - x0)

    base_x, base_y = points_xy[0]
    base_z = sample_terrain_z(TERRAIN, base_x, base_y) + calibration.z_offset_m + extra_z_m
    pose = actor.find("pose")
    if pose is not None:
        pose.text = _format_pose(base_x, base_y, base_z, yaw)

    trajectory = actor.find("script/trajectory")
    if trajectory is None:
        return
    for child in list(trajectory):
        trajectory.remove(child)

    elapsed = 0.0
    for idx, (x, y) in enumerate(points_xy):
        if idx > 0:
            px, py = points_xy[idx - 1]
            elapsed += math.hypot(x - px, y - py) / max(speed_mps, 0.1)
        z = sample_terrain_z(TERRAIN, x, y) + calibration.z_offset_m + extra_z_m
        wp = ET.SubElement(trajectory, "waypoint")
        ET.SubElement(wp, "time").text = f"{elapsed:.2f}"
        ET.SubElement(wp, "pose").text = _format_pose(x, y, z, yaw)

    if len(points_xy) > 1:
        elapsed += 2.0
        z = sample_terrain_z(TERRAIN, base_x, base_y) + calibration.z_offset_m + extra_z_m
        wp = ET.SubElement(trajectory, "waypoint")
        ET.SubElement(wp, "time").text = f"{elapsed:.2f}"
        ET.SubElement(wp, "pose").text = _format_pose(base_x, base_y, z, yaw)


def _set_actor_scale(actor: ET.Element, scale: float) -> None:
    skin_scale = actor.find("skin/scale")
    if skin_scale is not None:
        skin_scale.text = f"{scale:.2f}"
    anim_scale = actor.find("animation/scale")
    if anim_scale is not None:
        anim_scale.text = f"{scale:.2f}"


def _rewrite_include_pose(include: ET.Element, x: float, y: float, yaw: float, calibration_key: str) -> None:
    calibration = calibration_for_key(calibration_key)
    z = sample_terrain_z(TERRAIN, x, y) + calibration.z_offset_m
    pose = include.find("pose")
    if pose is not None:
        pose.text = _format_pose(x, y, z, yaw)


def _add_ground_plane(world: ET.Element) -> None:
    """Add a flat ground plane for simplified validation worlds."""
    gp = ET.SubElement(world, "model")
    gp.set("name", "ground_plane")
    ET.SubElement(gp, "static").text = "true"
    link = ET.SubElement(gp, "link")
    link.set("name", "link")
    col = ET.SubElement(link, "collision")
    col.set("name", "collision")
    geom_col = ET.SubElement(col, "geometry")
    plane_col = ET.SubElement(geom_col, "plane")
    ET.SubElement(plane_col, "normal").text = "0 0 1"
    ET.SubElement(plane_col, "size").text = "1 1"
    vis = ET.SubElement(link, "visual")
    vis.set("name", "visual")
    geom_vis = ET.SubElement(vis, "geometry")
    plane_vis = ET.SubElement(geom_vis, "plane")
    ET.SubElement(plane_vis, "normal").text = "0 0 1"
    ET.SubElement(plane_vis, "size").text = "500 500"
    mat = ET.SubElement(vis, "material")
    ET.SubElement(mat, "ambient").text = "0.8 0.8 0.8 1"
    ET.SubElement(mat, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(mat, "specular").text = "0.8 0.8 0.8 1"


def _apply_scenario_customization(world_name: str, world: ET.Element) -> None:
    if world_name == "1779343687303_isr_rural_simple_single_person":
        # Remove any leftover actor element (replaced by static model include)
        for actor in world.findall("actor"):
            world.remove(actor)
        # Add Walking person static model at route start
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "name").text = "person_walking_test_001"
        x0, y0 = CUSTOM_PERSON_ROUTE[0]
        z0 = sample_terrain_z(TERRAIN, x0, y0) + 1.0
        ET.SubElement(include, "pose").text = f"{x0:.2f} {y0:.2f} {z0:.2f} 0.0000 0.0000 0.0000"
        ET.SubElement(include, "uri").text = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Walking person"
        return

    if world_name == "1779343687303_isr_rural_simple_single_vehicle":
        # Replace Fuel pickup with local test_vehicle_box (guaranteed render)
        include = None
        for candidate in world.findall("include"):
            if candidate.findtext("name") == "vehicle_dynamic_pickup_002":
                include = candidate
                break
        if include is not None:
            vx, vy = 224.0, -350.0  # 4 m from spawn for strong detectability
            uri = include.find("uri")
            if uri is not None:
                uri.text = "model://test_vehicle_box"
            pose = include.find("pose")
            if pose is not None:
                # test_vehicle_box center is at model origin; half-height = 0.75 m
                pose.text = f"{vx:.2f} {vy:.2f} 0.80 0.0000 0.0000 0.0000"
        return

    if world_name == "1779343687303_isr_rural_simple_person_vehicle_separated":
        # Replace actor with proven static person model
        for actor in world.findall("actor"):
            world.remove(actor)
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "name").text = "person_walking_test_001"
        x0, y0 = CUSTOM_PERSON_ROUTE[0]
        z0 = sample_terrain_z(TERRAIN, x0, y0) + 1.0
        ET.SubElement(include, "pose").text = f"{x0:.2f} {y0:.2f} {z0:.2f} 0.0000 0.0000 0.0000"
        ET.SubElement(include, "uri").text = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Walking person"
        # Replace Fuel hatchback with local test_vehicle_box (guaranteed render)
        include = None
        for candidate in world.findall("include"):
            if candidate.findtext("name") == "vehicle_dynamic_hatchback_001":
                include = candidate
                break
        if include is not None:
            vx, vy = 222.0, -345.0  # closer to spawn for detectability
            uri = include.find("uri")
            if uri is not None:
                uri.text = "model://test_vehicle_box"
            pose = include.find("pose")
            if pose is not None:
                pose.text = f"{vx:.2f} {vy:.2f} 0.80 0.0000 0.0000 0.0000"
        return


def generate_worlds() -> list[Path]:
    src = PX4_WORLDS_DIR / SOURCE_WORLD
    if not src.exists():
        raise FileNotFoundError(src)

    root = ET.parse(src).getroot()
    world = root.find("world")
    if world is None:
        raise RuntimeError("Missing <world> in source world")

    outputs: list[Path] = []
    for world_name, spec in SCENARIOS.items():
        new_root = copy.deepcopy(root)
        new_world = new_root.find("world")
        assert new_world is not None
        new_world.set("name", world_name)
        for child in list(new_world):
            if not _keep_element(child, spec):
                new_world.remove(child)
        _apply_scenario_customization(world_name, new_world)
        _add_ground_plane(new_world)
        ET.indent(new_root, space="  ")
        out_path = PX4_WORLDS_DIR / f"{world_name}.sdf"
        out_path.write_text(
            '<?xml version="1.0" ?>\n' + ET.tostring(new_root, encoding="unicode"),
            encoding="utf-8",
        )
        outputs.append(out_path)
    return outputs


def main() -> None:
    paths = generate_worlds()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
