"""Generate the Stage 1C grounded dynamic ISR world.

Rebuilds the dynamic / human / vehicle layer on top of the validated
realistic-lite-v2 terrain and environmental scene, with:
  - broader target distribution across multiple interest zones
  - longer, smoother routes for sustained tracking tests
  - realistic Fuel vegetation (oak / pine trees) replacing inline geometric vegetation
  - increased human and vehicle counts with better spatial separation
  - preserved bilinear terrain sampling and per-asset contact offsets
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from apps.tools.place_real_terrain_vegetation import load_terrain

WORLD_ID = "1779343687303"
BASE_WORLD = f"{WORLD_ID}_isr_rural_realistic_lite_v2"
TARGET_WORLD = f"{WORLD_ID}_isr_rural_dynamic_v3"
PX4_WORLDS_DIR = PROJECT_ROOT.parent / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "worlds"

CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"
FUEL_BASE = "https://fuel.gazebosim.org/1.0/OpenRobotics/models"


@dataclass(frozen=True)
class AssetCalibration:
    key: str
    asset_type: str
    z_offset_m: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_offset_deg: float = 0.0
    contact_type: str = "base"
    floating_tol_m: float = 0.10
    penetration_tol_m: float = 0.10
    notes: str = ""
    approximate: bool = False


@dataclass(frozen=True)
class ActorAsset:
    name: str
    mesh_path: Path
    scale: float
    animation_type: str
    calibration_key: str
    model_uri: str | None = None


@dataclass(frozen=True)
class DynamicRoute:
    name: str
    category: str
    asset: ActorAsset
    waypoints_xy: list[tuple[float, float]]
    speeds_mps: list[float]
    pauses_s: list[float]
    sample_spacing_m: float


@dataclass(frozen=True)
class StaticIncludePlacement:
    name: str
    uri: str
    calibration_key: str
    x: float
    y: float
    yaw_deg: float


ASSET_CALIBRATIONS: dict[str, AssetCalibration] = {
    "mingfei_walk": AssetCalibration(
        key="mingfei_walk",
        asset_type="human",
        z_offset_m=1.00,
        contact_type="feet",
        floating_tol_m=0.10,
        penetration_tol_m=0.05,
        notes="Actor model.sdf uses a 1.0 m base pose; foot contact tracks terrain with that offset.",
    ),
    "femalevisitor": AssetCalibration(
        key="femalevisitor",
        asset_type="human",
        z_offset_m=0.02,
        contact_type="feet",
        floating_tol_m=0.10,
        penetration_tol_m=0.05,
        notes="OBJ base is nearly on Z=0; a slight lift avoids clipping into sampled terrain.",
        approximate=True,
    ),
    "walking_person_static": AssetCalibration(
        key="walking_person_static",
        asset_type="human",
        z_offset_m=0.04,
        contact_type="feet",
        floating_tol_m=0.10,
        penetration_tol_m=0.05,
        notes="Walking person visual already has a small -0.02 local pose; modest lift keeps shoes visible.",
        approximate=True,
    ),
    "malevisitoronphone": AssetCalibration(
        key="malevisitoronphone",
        asset_type="human",
        z_offset_m=0.03,
        contact_type="feet",
        floating_tol_m=0.10,
        penetration_tol_m=0.05,
        notes="Static OBJ stands upright; tiny lift prevents toe clipping.",
        approximate=True,
    ),
    "pickup": AssetCalibration(
        key="pickup",
        asset_type="vehicle",
        z_offset_m=0.10,
        contact_type="wheels",
        floating_tol_m=0.15,
        penetration_tol_m=0.10,
        notes="Pickup mesh sits low on uneven terrain; small base lift plus denser terrain sampling keeps wheels visible.",
        approximate=True,
    ),
    "truckbox": AssetCalibration(
        key="truckbox",
        asset_type="vehicle",
        z_offset_m=0.06,
        contact_type="wheels",
        floating_tol_m=0.15,
        penetration_tol_m=0.10,
        notes="TruckBox OBJ origin is near ground base but benefits from a small terrain clearance.",
        approximate=True,
    ),
    "hatchback": AssetCalibration(
        key="hatchback",
        asset_type="vehicle",
        z_offset_m=0.06,
        contact_type="wheels",
        floating_tol_m=0.15,
        penetration_tol_m=0.10,
        notes="Hatchback mesh origin near wheel base; small clearance for uneven terrain.",
        approximate=True,
    ),
    "shed": AssetCalibration(
        key="shed",
        asset_type="building",
        z_offset_m=0.00,
        contact_type="building_base",
        floating_tol_m=0.10,
        penetration_tol_m=0.10,
        notes="Procedural shed walls already rise from local Z=0.",
    ),
    "tree": AssetCalibration(
        key="tree",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="trunk_base",
        floating_tol_m=0.15,
        penetration_tol_m=0.15,
        notes="Inline tree trunks start at local Z=0.",
    ),
    "bush": AssetCalibration(
        key="bush",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="shrub_base",
        floating_tol_m=0.15,
        penetration_tol_m=0.15,
        notes="Inline bush canopy rests on model origin.",
    ),
    "shrub": AssetCalibration(
        key="shrub",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="shrub_base",
        floating_tol_m=0.15,
        penetration_tol_m=0.15,
        notes="Inline shrub cylinder rises from model origin.",
    ),
    "grass": AssetCalibration(
        key="grass",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="shrub_base",
        floating_tol_m=0.15,
        penetration_tol_m=0.15,
        notes="Inline grass clump starts at model origin.",
    ),
    "rock": AssetCalibration(
        key="rock",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="base",
        floating_tol_m=0.15,
        penetration_tol_m=0.15,
        notes="Inline rocks are authored from local Z=0.",
    ),
    "concealment_zone": AssetCalibration(
        key="concealment_zone",
        asset_type="marker",
        z_offset_m=0.025,
        contact_type="base",
        floating_tol_m=0.15,
        penetration_tol_m=0.15,
        notes="Cylinder marker is centered on origin, so half-length must sit above ground.",
    ),
    "road": AssetCalibration(
        key="road",
        asset_type="road",
        z_offset_m=0.02,
        contact_type="base",
        floating_tol_m=0.20,
        penetration_tol_m=0.20,
        notes="Road strips intentionally hover slightly above terrain to avoid Z-fighting.",
    ),
    "oak_tree": AssetCalibration(
        key="oak_tree",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="trunk_base",
        floating_tol_m=0.20,
        penetration_tol_m=0.15,
        notes="Fuel Oak tree model origin at trunk base.",
    ),
    "pine_tree": AssetCalibration(
        key="pine_tree",
        asset_type="vegetation",
        z_offset_m=0.00,
        contact_type="trunk_base",
        floating_tol_m=0.20,
        penetration_tol_m=0.15,
        notes="Fuel Pine tree model origin at trunk base.",
    ),
}


STATIC_TARGETS: list[StaticIncludePlacement] = [
    StaticIncludePlacement(
        name="civilian_static_female_000",
        uri=f"{FUEL_BASE}/FemaleVisitor",
        calibration_key="femalevisitor",
        x=225.0,
        y=-345.0,
        yaw_deg=180.0,
    ),
    StaticIncludePlacement(
        name="civilian_static_phone_001",
        uri=f"{FUEL_BASE}/MaleVisitorOnPhone",
        calibration_key="malevisitoronphone",
        x=245.0,
        y=-375.0,
        yaw_deg=200.0,
    ),
    StaticIncludePlacement(
        name="civilian_static_walk_002",
        uri=f"{FUEL_BASE}/Walking person",
        calibration_key="walking_person_static",
        x=215.0,
        y=-335.0,
        yaw_deg=160.0,
    ),
    StaticIncludePlacement(
        name="civilian_static_phone_003",
        uri=f"{FUEL_BASE}/MaleVisitorOnPhone",
        calibration_key="malevisitoronphone",
        x=265.0,
        y=-355.0,
        yaw_deg=120.0,
    ),
    StaticIncludePlacement(
        name="vehicle_static_pickup_000",
        uri=f"{FUEL_BASE}/Pickup",
        calibration_key="pickup",
        x=230.0,
        y=-380.0,
        yaw_deg=90.0,
    ),
    StaticIncludePlacement(
        name="vehicle_static_truckbox_001",
        uri=f"{FUEL_BASE}/TruckBox",
        calibration_key="truckbox",
        x=250.0,
        y=-330.0,
        yaw_deg=45.0,
    ),
    StaticIncludePlacement(
        name="vehicle_static_hatchback_002",
        uri=f"{FUEL_BASE}/Hatchback",
        calibration_key="hatchback",
        x=270.0,
        y=-365.0,
        yaw_deg=135.0,
    ),
]


# Realistic Fuel tree placements: (x, y, yaw_deg, type)
# Types: "oak" or "pine". Scattered across a broad zone, avoiding spawn.
TREE_PLACEMENTS: list[tuple[float, float, float, str]] = [
    # Near north ridge
    (215.0, -305.0, 15.0, "oak"),
    (235.0, -298.0, 45.0, "pine"),
    (255.0, -308.0, 0.0, "oak"),
    (275.0, -302.0, 75.0, "pine"),
    # East slope
    (285.0, -340.0, 30.0, "oak"),
    (295.0, -360.0, 10.0, "pine"),
    (290.0, -380.0, 60.0, "oak"),
    # South valley
    (205.0, -395.0, 20.0, "pine"),
    (225.0, -405.0, 0.0, "oak"),
    (245.0, -395.0, 40.0, "pine"),
    # West approach
    (175.0, -325.0, 55.0, "oak"),
    (165.0, -355.0, 25.0, "pine"),
    (185.0, -375.0, 0.0, "oak"),
    # Mid / spawn perimeter
    (210.0, -365.0, 35.0, "pine"),
    (240.0, -345.0, 15.0, "oak"),
    (260.0, -385.0, 50.0, "pine"),
]


def _cached_path(*parts: str) -> Path:
    path = CACHE_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Required cached asset missing: {path}")
    return path


def _mesh_path(path: Path) -> str:
    return str(path.resolve())


MINGFEI_WALK = ActorAsset(
    name="mingfei_walk",
    mesh_path=_cached_path("mingfei", "models", "actor", "1", "meshes", "walk.dae"),
    scale=1.00,
    animation_type="walk",
    calibration_key="mingfei_walk",
)

PICKUP_MESH = ActorAsset(
    name="pickup",
    mesh_path=_cached_path("openrobotics", "models", "pickup", "4", "meshes", "pickup.dae"),
    scale=1.0,
    animation_type="drive",
    calibration_key="pickup",
    model_uri=f"{FUEL_BASE}/Pickup",
)

TRUCKBOX_MESH = ActorAsset(
    name="truckbox",
    mesh_path=_cached_path("openrobotics", "models", "truckbox", "2", "meshes", "truck.obj"),
    scale=1.0,
    animation_type="drive",
    calibration_key="truckbox",
    model_uri=f"{FUEL_BASE}/TruckBox",
)

HATCHBACK_MESH = ActorAsset(
    name="hatchback",
    mesh_path=_cached_path("openrobotics", "models", "hatchback", "3", "meshes", "hatchback.obj"),
    scale=1.0,
    animation_type="drive",
    calibration_key="hatchback",
    model_uri=f"{FUEL_BASE}/Hatchback",
)


def sample_terrain_z(terrain, world_x: float, world_y: float) -> float:
    """Bilinear terrain sampler for smoother grounding than nearest-pixel lookup."""
    if not terrain.pixels:
        return terrain.offset_z

    local_x = world_x - terrain.offset_x + terrain.width_m / 2
    local_y = world_y - terrain.offset_y + terrain.height_m / 2
    px = max(0.0, min(local_x / terrain.width_m * (terrain.img_w - 1), terrain.img_w - 1))
    py = max(0.0, min(local_y / terrain.height_m * (terrain.img_h - 1), terrain.img_h - 1))

    x0 = int(math.floor(px))
    x1 = min(x0 + 1, terrain.img_w - 1)
    y0 = int(math.floor(py))
    y1 = min(y0 + 1, terrain.img_h - 1)
    fx = px - x0
    fy = py - y0

    p = terrain.pixels
    v00 = p[y0][x0]
    v10 = p[y0][x1]
    v01 = p[y1][x0]
    v11 = p[y1][x1]
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    v = v0 * (1.0 - fy) + v1 * fy
    return terrain.offset_z + v * terrain.elevation_m


def terrain_slope_deg(terrain, world_x: float, world_y: float, step_m: float = 0.5) -> float:
    zx1 = sample_terrain_z(terrain, world_x + step_m, world_y)
    zx0 = sample_terrain_z(terrain, world_x - step_m, world_y)
    zy1 = sample_terrain_z(terrain, world_x, world_y + step_m)
    zy0 = sample_terrain_z(terrain, world_x, world_y - step_m)
    dzdx = (zx1 - zx0) / (2.0 * step_m)
    dzdy = (zy1 - zy0) / (2.0 * step_m)
    return math.degrees(math.atan(math.hypot(dzdx, dzdy)))


def _parse_pose(pose_text: str | None) -> tuple[float, float, float, float, float, float]:
    if not pose_text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = [float(v) for v in pose_text.strip().split()[:6]]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals)  # type: ignore[return-value]


def _format_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> str:
    return f"{x:.2f} {y:.2f} {z:.2f} {roll:.4f} {pitch:.4f} {yaw:.4f}"


def _heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def calibration_for_key(key: str) -> AssetCalibration:
    return ASSET_CALIBRATIONS[key]


def calibration_for_include(name: str, uri: str) -> AssetCalibration:
    uri_l = uri.lower()
    name_l = name.lower()
    if "femalevisitor" in uri_l or "femalevisitor" in name_l:
        return calibration_for_key("femalevisitor")
    if "walking person" in uri_l or "walking_person" in name_l:
        return calibration_for_key("walking_person_static")
    if "malevisitoronphone" in uri_l or "malevisitoronphone" in name_l:
        return calibration_for_key("malevisitoronphone")
    if "pickup" in uri_l or "pickup" in name_l:
        return calibration_for_key("pickup")
    if "truckbox" in uri_l or "truckbox" in name_l:
        return calibration_for_key("truckbox")
    if "hatchback" in uri_l or "hatchback" in name_l:
        return calibration_for_key("hatchback")
    if "oak" in uri_l or "oak" in name_l:
        return calibration_for_key("oak_tree")
    if "pine" in uri_l or "pine" in name_l:
        return calibration_for_key("pine_tree")
    raise KeyError(f"No include calibration for {name} / {uri}")


def calibration_for_model(name: str) -> AssetCalibration:
    if name.startswith("shed_"):
        return calibration_for_key("shed")
    if name.startswith(("veg_tree_", "tl_tree_")):
        return calibration_for_key("tree")
    if name.startswith(("veg_bush_", "tl_bush_")):
        return calibration_for_key("bush")
    if name.startswith(("veg_shrub_", "tl_shrub_")):
        return calibration_for_key("shrub")
    if name.startswith(("veg_grass_",)):
        return calibration_for_key("grass")
    if name.startswith(("veg_rock_",)):
        return calibration_for_key("rock")
    if name.startswith("concealment_zone_"):
        return calibration_for_key("concealment_zone")
    if name.startswith("road_"):
        return calibration_for_key("road")
    raise KeyError(f"No model calibration for {name}")


def classify_entity(name: str, tag: str, uri: str | None = None) -> AssetCalibration | None:
    try:
        if tag == "include" and uri is not None:
            return calibration_for_include(name, uri)
        if tag == "model":
            return calibration_for_model(name)
        if tag == "actor":
            return calibration_for_key("mingfei_walk")
    except KeyError:
        return None
    return None


def pose_for_xy(
    terrain,
    x: float,
    y: float,
    yaw_rad: float,
    calibration: AssetCalibration,
) -> tuple[float, float, float, float, float, float]:
    z = sample_terrain_z(terrain, x, y) + calibration.z_offset_m
    roll = math.radians(calibration.roll_deg)
    pitch = math.radians(calibration.pitch_deg)
    yaw = yaw_rad + math.radians(calibration.yaw_offset_deg)
    return x, y, z, roll, pitch, yaw


def _expanded_route_schedule(route: DynamicRoute, terrain) -> list[tuple[float, float, float, float, float]]:
    """Densify route waypoints so motion samples terrain height along the path."""
    calibration = calibration_for_key(route.asset.calibration_key)
    schedule: list[tuple[float, float, float, float, float]] = []
    t_curr = 0.0
    points = route.waypoints_xy

    for idx, start in enumerate(points):
        end = points[(idx + 1) % len(points)]
        seg_yaw = _heading(start, end)
        seg_len = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(math.ceil(seg_len / route.sample_spacing_m)))
        speed = max(route.speeds_mps[idx], 0.1)

        for step_idx in range(steps):
            frac = step_idx / steps
            x = start[0] + (end[0] - start[0]) * frac
            y = start[1] + (end[1] - start[1]) * frac
            _, _, z, _, _, yaw = pose_for_xy(terrain, x, y, seg_yaw, calibration)
            schedule.append((t_curr, x, y, z, yaw))
            if step_idx < steps:
                t_curr += (seg_len / steps) / speed

        # Pause at the authored waypoint, not at every densified sample.
        _, _, z_end, _, _, yaw_end = pose_for_xy(terrain, end[0], end[1], seg_yaw, calibration)
        pause_s = route.pauses_s[idx]
        if pause_s > 0.0:
            schedule.append((t_curr, end[0], end[1], z_end, yaw_end))
            t_curr += pause_s

    first = points[0]
    second = points[1] if len(points) > 1 else points[0]
    first_yaw = _heading(first, second)
    _, _, first_z, _, _, first_pose_yaw = pose_for_xy(terrain, first[0], first[1], first_yaw, calibration)
    schedule.append((t_curr, first[0], first[1], first_z, first_pose_yaw))
    return schedule


def _remove_children(world: ET.Element, predicate) -> None:
    for child in [c for c in list(world) if predicate(c)]:
        world.remove(child)


def _include_name(elem: ET.Element) -> str:
    return (elem.findtext("name") or "").strip()


def _model_name(elem: ET.Element) -> str:
    return elem.get("name", "")


def _prune_baseline(world: ET.Element) -> None:
    def should_remove(child: ET.Element) -> bool:
        tag = child.tag.split("}")[-1]
        if tag == "include":
            name = _include_name(child)
            return (
                name.startswith("civilian_")
                or name.startswith("military_")
                or name.startswith("vehicle_")
                or name.startswith("calib_")
            )
        if tag == "model":
            name = _model_name(child)
            return (
                name.startswith("marker_")
                or name.startswith("actor_")
                or name.startswith("veg_")
                or name.startswith("tl_")
            )
        if tag == "actor":
            return True
        return False

    _remove_children(world, should_remove)


def _retarget_grounded_models(world: ET.Element, terrain) -> None:
    for model in world.findall("./model"):
        name = model.get("name", "")
        calibration = classify_entity(name, "model")
        if calibration is None:
            continue
        x, y, _z, _r, _p, yaw = _parse_pose(model.findtext("pose"))
        pose_el = model.find("pose")
        if pose_el is None:
            pose_el = ET.SubElement(model, "pose")
        pose_el.text = _format_pose(*pose_for_xy(terrain, x, y, yaw, calibration))


def _add_static_include(world: ET.Element, terrain, placement: StaticIncludePlacement) -> None:
    calibration = calibration_for_key(placement.calibration_key)
    include = ET.SubElement(world, "include")
    ET.SubElement(include, "name").text = placement.name
    x, y, z, roll, pitch, yaw = pose_for_xy(
        terrain,
        placement.x,
        placement.y,
        math.radians(placement.yaw_deg),
        calibration,
    )
    ET.SubElement(include, "pose").text = _format_pose(x, y, z, roll, pitch, yaw)
    ET.SubElement(include, "uri").text = placement.uri


def _add_human_actor(world: ET.Element, route: DynamicRoute, terrain) -> None:
    schedule = _expanded_route_schedule(route, terrain)
    _, x0, y0, z0, yaw0 = schedule[0]

    actor = ET.SubElement(world, "actor", name=route.name)
    ET.SubElement(actor, "pose").text = _format_pose(x0, y0, z0, 0.0, 0.0, yaw0)

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
        ET.SubElement(wp, "time").text = f"{t_s:.2f}"
        ET.SubElement(wp, "pose").text = _format_pose(x, y, z, 0.0, 0.0, yaw)


def _add_vehicle_include(world: ET.Element, route: DynamicRoute, terrain) -> None:
    if route.asset.model_uri is None:
        raise ValueError(f"Vehicle route {route.name} missing model URI")
    calibration = calibration_for_key(route.asset.calibration_key)
    x0, y0 = route.waypoints_xy[0]
    yaw0 = _heading(route.waypoints_xy[0], route.waypoints_xy[1])
    include = ET.SubElement(world, "include")
    ET.SubElement(include, "name").text = route.name
    ET.SubElement(include, "pose").text = _format_pose(*pose_for_xy(terrain, x0, y0, yaw0, calibration))
    ET.SubElement(include, "uri").text = route.asset.model_uri


def _add_tree_include(world: ET.Element, terrain, x: float, y: float, yaw_deg: float, tree_type: str) -> None:
    calibration = calibration_for_key(f"{tree_type}_tree")
    include = ET.SubElement(world, "include")
    name = f"veg_{tree_type}_tree_{int(x)}_{int(abs(y))}"
    ET.SubElement(include, "name").text = name
    x, y, z, roll, pitch, yaw = pose_for_xy(
        terrain, x, y, math.radians(yaw_deg), calibration,
    )
    ET.SubElement(include, "pose").text = _format_pose(x, y, z, roll, pitch, yaw)
    tree_uri = f"{FUEL_BASE}/{'Oak tree' if tree_type == 'oak' else 'Pine tree'}"
    ET.SubElement(include, "uri").text = tree_uri


def _dynamic_routes() -> list[DynamicRoute]:
    return [
        # ------------------------------------------------------------------
        # Humans (10 moving Mingfei actors)
        # ------------------------------------------------------------------
        DynamicRoute(
            name="actor_north_patrol",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(220.0, -305.0), (260.0, -305.0), (260.0, -315.0), (220.0, -315.0)],
            speeds_mps=[1.0, 1.0, 1.0, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_valley_walk",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(200.0, -405.0), (240.0, -405.0), (240.0, -395.0), (200.0, -395.0)],
            speeds_mps=[1.2, 1.0, 1.2, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_east_ridge",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(250.0, -340.0), (290.0, -360.0), (280.0, -380.0), (260.0, -370.0), (250.0, -340.0)],
            speeds_mps=[1.0, 0.9, 1.0, 0.9, 1.0],
            pauses_s=[0.0, 1.0, 0.0, 1.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_west_trail",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(180.0, -330.0), (200.0, -350.0), (190.0, -370.0), (170.0, -350.0), (180.0, -330.0)],
            speeds_mps=[1.0, 1.0, 1.0, 1.0, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_crossing_main",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(210.0, -355.0), (235.0, -355.0)],
            speeds_mps=[1.2, 1.2],
            pauses_s=[2.0, 2.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_south_loop",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(210.0, -395.0), (230.0, -385.0), (240.0, -395.0), (220.0, -405.0), (210.0, -395.0)],
            speeds_mps=[1.0, 1.0, 1.0, 1.0, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_diagonal_long",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(190.0, -380.0), (240.0, -320.0), (250.0, -330.0), (200.0, -390.0), (190.0, -380.0)],
            speeds_mps=[1.1, 1.0, 1.1, 1.0, 1.1],
            pauses_s=[0.0, 1.5, 0.0, 1.5, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_east_loop",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(270.0, -350.0), (290.0, -340.0), (295.0, -360.0), (275.0, -370.0), (270.0, -350.0)],
            speeds_mps=[1.0, 1.0, 1.0, 1.0, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_west_loop",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(170.0, -360.0), (185.0, -340.0), (175.0, -320.0), (165.0, -340.0), (170.0, -360.0)],
            speeds_mps=[1.0, 1.0, 1.0, 1.0, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        DynamicRoute(
            name="actor_mid_patrol",
            category="human",
            asset=MINGFEI_WALK,
            waypoints_xy=[(230.0, -340.0), (250.0, -340.0), (250.0, -360.0), (230.0, -360.0), (230.0, -340.0)],
            speeds_mps=[1.0, 1.0, 1.0, 1.0, 1.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.0,
        ),
        # ------------------------------------------------------------------
        # Vehicles (5 moving)
        # ------------------------------------------------------------------
        DynamicRoute(
            name="vehicle_dynamic_pickup_001",
            category="vehicle",
            asset=PICKUP_MESH,
            waypoints_xy=[
                (210.0, -365.0), (235.0, -375.0), (245.0, -385.0),
                (220.0, -395.0), (200.0, -385.0), (210.0, -365.0),
            ],
            speeds_mps=[2.5, 2.0, 2.2, 2.0, 2.5, 2.5],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.5,
        ),
        DynamicRoute(
            name="vehicle_dynamic_truckbox_001",
            category="vehicle",
            asset=TRUCKBOX_MESH,
            waypoints_xy=[
                (230.0, -340.0), (260.0, -350.0), (270.0, -370.0),
                (240.0, -380.0), (220.0, -370.0), (230.0, -340.0),
            ],
            speeds_mps=[1.8, 1.5, 1.8, 1.5, 1.8, 1.8],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.5,
        ),
        DynamicRoute(
            name="vehicle_dynamic_hatchback_001",
            category="vehicle",
            asset=HATCHBACK_MESH,
            waypoints_xy=[
                (180.0, -330.0), (200.0, -320.0), (220.0, -330.0),
                (200.0, -340.0), (180.0, -330.0),
            ],
            speeds_mps=[3.0, 3.5, 3.0, 3.5, 3.0],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.5,
        ),
        DynamicRoute(
            name="vehicle_dynamic_pickup_002",
            category="vehicle",
            asset=PICKUP_MESH,
            waypoints_xy=[
                (250.0, -355.0), (280.0, -365.0), (290.0, -385.0),
                (260.0, -395.0), (240.0, -385.0), (250.0, -355.0),
            ],
            speeds_mps=[2.5, 2.0, 2.2, 2.0, 2.5, 2.5],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.5,
        ),
        DynamicRoute(
            name="vehicle_dynamic_truckbox_002",
            category="vehicle",
            asset=TRUCKBOX_MESH,
            waypoints_xy=[
                (190.0, -350.0), (210.0, -330.0), (230.0, -340.0),
                (210.0, -360.0), (190.0, -350.0),
            ],
            speeds_mps=[1.8, 1.5, 1.8, 1.5, 1.8, 1.8],
            pauses_s=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_spacing_m=1.5,
        ),
    ]


def dynamic_vehicle_route_specs() -> list[dict]:
    """Shared route bundle for run_gazebo_traffic and audit tooling."""
    return [
        {
            "model_name": route.name,
            "calibration_key": route.asset.calibration_key,
            "waypoints_xy": route.waypoints_xy,
            "speeds_mps": route.speeds_mps,
            "pauses_s": route.pauses_s,
            "sample_spacing_m": route.sample_spacing_m,
        }
        for route in _dynamic_routes()
        if route.category == "vehicle"
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
    _retarget_grounded_models(world, terrain)

    for placement in STATIC_TARGETS:
        _add_static_include(world, terrain, placement)

    for route in _dynamic_routes():
        if route.category == "human":
            _add_human_actor(world, route, terrain)
        else:
            _add_vehicle_include(world, route, terrain)

    for tx, ty, tyaw, ttype in TREE_PLACEMENTS:
        _add_tree_include(world, terrain, tx, ty, tyaw, ttype)

    ET.indent(root, space="  ")
    target_sdf.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return target_sdf


def main() -> None:
    target_sdf = generate_world()
    print(f"Wrote {target_sdf}")
    print("Dynamic layer summary:")
    print("  Humans: 10 moving (Mingfei) + 4 static")
    print("  Vehicles: 5 moving + 3 static")
    print("  Vegetation: 16 Fuel trees (oak + pine) replacing inline geometric vegetation")
    print("  Spatial envelope: x=160-300, y=-320 to -410 (~140x90m zone)")
    print("  Interest zones: north ridge, south valley, east slope, west approach, mid corridor")
    print("  Grounding: bilinear terrain sampling + shared contact calibration table")
    print("  Bad assets removed: Rescue Randy, baseline calibration dummies, MaleVisitorWalk, inline veg")


if __name__ == "__main__":
    main()
