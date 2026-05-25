"""Generate the Phase 6 dynamic ISR world from the validated realistic-lite-v2 base.

This keeps the validated terrain, camera, and baseline placement work intact while
rebuilding the dynamic layer to be operationally realistic enough for:
  - moving human tracking
  - partial occlusion and reacquisition
  - slow moving vehicle tracking
  - safe standoff / aim-lock evaluation

The generator deliberately does not modify any baseline world. It only writes:
  PX4-Autopilot/Tools/simulation/gz/worlds/1779343687303_isr_rural_dynamic_v3.sdf
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from apps.tools.place_real_terrain_vegetation import load_terrain, terrain_z

WORLD_ID = "1779343687303"
BASE_WORLD = f"{WORLD_ID}_isr_rural_realistic_lite_v2"
TARGET_WORLD = f"{WORLD_ID}_isr_rural_dynamic_v3"
PX4_WORLDS_DIR = PROJECT_ROOT.parent / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "worlds"

CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"


@dataclass(frozen=True)
class ActorAsset:
    name: str
    mesh_path: Path
    scale: float
    z_offset: float
    animation_type: str
    model_uri: str | None = None


@dataclass(frozen=True)
class DynamicRoute:
    name: str
    category: str
    asset: ActorAsset
    waypoints_xy: list[tuple[float, float]]
    speeds_mps: list[float]
    pauses_s: list[float]


def _mesh_path(path: Path) -> str:
    # Gazebo Harmonic fails to resolve percent-encoded file:// URIs for cached
    # Fuel assets with spaces in their directory names (for example
    # "male visitor"). Use plain absolute filesystem paths instead.
    return str(path.resolve())


def _cached_path(*parts: str) -> Path:
    path = CACHE_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Required cached asset missing: {path}")
    return path


MALE_VISITOR_WALK = ActorAsset(
    name="male_visitor_walk",
    mesh_path=_cached_path("openrobotics", "models", "male visitor", "2", "meshes", "MaleVisitorWalk.dae"),
    scale=1.02,
    z_offset=1.0,
    animation_type="walking",
)

MINGFEI_WALK = ActorAsset(
    name="mingfei_walk",
    mesh_path=_cached_path("mingfei", "models", "actor", "1", "meshes", "walk.dae"),
    scale=1.00,
    z_offset=1.0,
    animation_type="walk",
)

PICKUP_MESH = ActorAsset(
    name="pickup",
    mesh_path=_cached_path("openrobotics", "models", "pickup", "4", "meshes", "pickup.dae"),
    scale=1.0,
    z_offset=0.0,
    animation_type="drive",
    model_uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/Pickup",
)

TRUCKBOX_MESH = ActorAsset(
    name="truckbox",
    mesh_path=_cached_path("openrobotics", "models", "truckbox", "2", "meshes", "truck.obj"),
    scale=1.0,
    z_offset=0.0,
    animation_type="drive",
    model_uri="https://fuel.gazebosim.org/1.0/OpenRobotics/models/TruckBox",
)


KEEP_STATIC_HUMANS = {
    "civilian_standing_000",
    "civilian_standing_002",
    "civilian_standing_005",
    "military_static_001",
}

KEEP_STATIC_VEHICLES = {
    "vehicle_civ_pickup_000",
    "vehicle_concealed_000",
}


def _heading_to_next(points: list[tuple[float, float]], idx: int) -> float:
    curr_x, curr_y = points[idx]
    next_x, next_y = points[(idx + 1) % len(points)]
    return math.atan2(next_y - curr_y, next_x - curr_x)


def _waypoint_schedule(route: DynamicRoute, terrain) -> list[tuple[float, float, float, float, float]]:
    schedule: list[tuple[float, float, float, float, float]] = []
    t_curr = 0.0
    for idx, (x, y) in enumerate(route.waypoints_xy):
        yaw = _heading_to_next(route.waypoints_xy, idx)
        z = terrain_z(terrain, x, y) + route.asset.z_offset
        schedule.append((t_curr, x, y, z, yaw))

        pause_s = route.pauses_s[idx]
        if pause_s > 0.0:
            t_curr += pause_s
            schedule.append((t_curr, x, y, z, yaw))

        next_x, next_y = route.waypoints_xy[(idx + 1) % len(route.waypoints_xy)]
        segment_len = math.hypot(next_x - x, next_y - y)
        segment_speed = max(route.speeds_mps[idx], 0.1)
        t_curr += segment_len / segment_speed

    first_x, first_y = route.waypoints_xy[0]
    first_z = terrain_z(terrain, first_x, first_y) + route.asset.z_offset
    first_yaw = _heading_to_next(route.waypoints_xy, 0)
    schedule.append((t_curr, first_x, first_y, first_z, first_yaw))
    return schedule


def _remove_children(world: ET.Element, predicate) -> None:
    to_remove = [child for child in list(world) if predicate(child)]
    for child in to_remove:
        world.remove(child)


def _include_name(elem: ET.Element) -> str:
    name_el = elem.find("name")
    return name_el.text.strip() if name_el is not None and name_el.text else ""


def _actor_name(elem: ET.Element) -> str:
    return elem.get("name", "")


def _model_name(elem: ET.Element) -> str:
    return elem.get("name", "")


def _prune_baseline(world: ET.Element) -> None:
    def should_remove(child: ET.Element) -> bool:
        tag = child.tag.split("}")[-1]
        if tag == "include":
            name = _include_name(child)
            if name.startswith("civilian_standing_") and name not in KEEP_STATIC_HUMANS:
                return True
            if name.startswith("military_static_") and name not in KEEP_STATIC_HUMANS:
                return True
            if name.startswith("vehicle_") and name not in KEEP_STATIC_VEHICLES:
                return True
        elif tag == "model":
            name = _model_name(child)
            if name.startswith("marker_civilian_standing_"):
                return name.removeprefix("marker_") not in KEEP_STATIC_HUMANS
            if name.startswith("marker_military_static_"):
                return name.removeprefix("marker_") not in KEEP_STATIC_HUMANS
            if name.startswith("marker_vehicle_"):
                return name.removeprefix("marker_") not in KEEP_STATIC_VEHICLES
            if name.startswith("actor_"):
                return True
        elif tag == "actor":
            return True
        return False

    _remove_children(world, should_remove)


def _add_human_actor(world: ET.Element, route: DynamicRoute, terrain) -> None:
    schedule = _waypoint_schedule(route, terrain)
    _, x0, y0, z0, yaw0 = schedule[0]

    actor = ET.SubElement(world, "actor", name=route.name)
    ET.SubElement(actor, "pose").text = f"{x0:.2f} {y0:.2f} {z0:.2f} 0 0 {yaw0:.2f}"

    skin = ET.SubElement(actor, "skin")
    ET.SubElement(skin, "filename").text = _mesh_path(route.asset.mesh_path)
    ET.SubElement(skin, "scale").text = f"{route.asset.scale:.2f}"

    animation = ET.SubElement(actor, "animation", name=route.asset.animation_type)
    ET.SubElement(animation, "filename").text = _mesh_path(route.asset.mesh_path)
    ET.SubElement(animation, "scale").text = f"{route.asset.scale:.2f}"
    ET.SubElement(animation, "interpolate_x").text = "true"

    script = ET.SubElement(actor, "script")
    ET.SubElement(script, "loop").text = "true"
    ET.SubElement(script, "auto_start").text = "true"
    trajectory = ET.SubElement(script, "trajectory", id="0", type=route.asset.animation_type)

    for t_s, x, y, z, yaw in schedule:
        wp = ET.SubElement(trajectory, "waypoint")
        ET.SubElement(wp, "time").text = f"{t_s:.1f}"
        ET.SubElement(wp, "pose").text = f"{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.2f}"


def _add_vehicle_include(world: ET.Element, route: DynamicRoute, terrain) -> None:
    schedule = _waypoint_schedule(route, terrain)
    _, x0, y0, z0, yaw0 = schedule[0]

    if route.asset.model_uri is None:
        raise ValueError(f"Vehicle route {route.name} missing model URI")

    include = ET.SubElement(world, "include")
    ET.SubElement(include, "name").text = route.name
    ET.SubElement(include, "pose").text = f"{x0:.2f} {y0:.2f} {z0:.2f} 0 0 {yaw0:.2f}"
    ET.SubElement(include, "uri").text = route.asset.model_uri


def _dynamic_routes() -> list[DynamicRoute]:
    return [
        DynamicRoute(
            name="actor_road_walker",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(220.0, -350.0), (228.0, -393.0), (215.0, -391.0), (208.0, -388.0)],
            speeds_mps=[1.2, 1.0, 1.1, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0],
        ),
        DynamicRoute(
            name="actor_tree_walker",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(205.0, -345.0), (202.0, -352.0), (198.0, -360.0), (192.0, -371.0), (202.0, -379.0)],
            speeds_mps=[0.9, 1.0, 1.0, 0.9, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        DynamicRoute(
            name="actor_shed_walker",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(216.0, -350.0), (221.0, -348.0), (219.5, -347.3)],
            speeds_mps=[0.9, 0.8, 0.9],
            pauses_s=[3.0, 4.0, 2.0],
        ),
        DynamicRoute(
            name="actor_crossing",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(210.0, -360.0), (235.0, -360.0)],
            speeds_mps=[1.3, 1.1],
            pauses_s=[0.0, 0.0],
        ),
        DynamicRoute(
            name="vehicle_dynamic_pickup_001",
            category="vehicle",
            asset=PICKUP_MESH,
            waypoints_xy=[(220.0, -350.0), (228.0, -393.0), (215.0, -391.0), (208.0, -388.0)],
            speeds_mps=[3.2, 2.6, 2.2, 2.8],
            pauses_s=[1.5, 0.0, 0.0, 0.0],
        ),
        DynamicRoute(
            name="vehicle_dynamic_truckbox_001",
            category="vehicle",
            asset=TRUCKBOX_MESH,
            waypoints_xy=[(198.0, -379.0), (205.0, -383.0), (214.0, -389.0), (223.0, -394.0)],
            speeds_mps=[1.8, 2.1, 1.9, 2.2],
            pauses_s=[3.0, 1.5, 2.0, 4.0],
        ),
    ]


def generate_world() -> Path:
    base_sdf = PX4_WORLDS_DIR / f"{BASE_WORLD}.sdf"
    target_sdf = PX4_WORLDS_DIR / f"{TARGET_WORLD}.sdf"

    terrain = load_terrain(WORLD_ID)
    tree = ET.parse(base_sdf)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise RuntimeError("Missing <world> element")

    world.set("name", TARGET_WORLD)
    _prune_baseline(world)

    for route in _dynamic_routes():
        if route.category == "human":
            _add_human_actor(world, route, terrain)
        else:
            _add_vehicle_include(world, route, terrain)

    ET.indent(root, space="  ")
    target_sdf.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return target_sdf


def main() -> None:
    target_sdf = generate_world()
    print(f"Wrote {target_sdf}")
    print("Dynamic layer summary:")
    print("  Humans: 4 moving + 4 static")
    print("  Vehicles: 2 moving + 2 static")
    print("  Moving patterns: road walker, tree-line walker, shed-to-vehicle walker, crossing target, patrol pickup, stop-go truck")
    print("  Assets: cached local Fuel meshes only for dynamic actors")


if __name__ == "__main__":
    main()
