"""Place realistic Fuel mesh people and vehicles on real-terrain Gazebo worlds.

Uses cached Fuel models (Hatchback, Bus, Male Visitor, FemaleVisitor, etc.)
instead of inline primitive geometry for COCO-realistic YOLO detection targets.
Vegetation and terrain infrastructure are reused from place_real_terrain_vegetation.

Usage:
    python -m apps.tools.place_realistic_targets --world 1779343687303 --density light
    python -m apps.tools.place_realistic_targets --world 1779343687303 --all
"""

from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from apps.tools.place_real_terrain_vegetation import (
    DENSITY_PROFILES,
    PERSON_DEFS,
    VEHICLE_DEFS,
    TerrainInfo,
    _add_concealment_markers,
    _add_inline_vegetation,
    _sdf_header,
    audit_placements,
    generate_placements,
    load_terrain,
    terrain_z,
)
from apps.tools.import_fuel_assets import (
    PEOPLE_MODELS,
    VEHICLE_MODELS,
    FuelModelInfo,
    verify_all,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PX4_DIR = PROJECT_ROOT.parent / "PX4-Autopilot"
GZ_WORLDS_DIR = PX4_DIR / "Tools" / "simulation" / "gz" / "worlds"

FUEL_BASE = "https://fuel.gazebosim.org/1.0"

# ---------------------------------------------------------------------------
# Fuel model URIs
# ---------------------------------------------------------------------------

FUEL_VEHICLES = [
    {"key": "hatchback", "uri": f"{FUEL_BASE}/OpenRobotics/models/Hatchback", "z_off": 0.0},
    {"key": "pickup", "uri": f"{FUEL_BASE}/OpenRobotics/models/Pickup", "z_off": 0.0},
    {"key": "truckbox", "uri": f"{FUEL_BASE}/OpenRobotics/models/TruckBox", "z_off": 0.0},
    {"key": "bus", "uri": f"{FUEL_BASE}/OpenRobotics/models/Bus", "z_off": 0.0},
]

# Walking actor skins — rotate for visual variety
ACTOR_SKINS = [
    f"{FUEL_BASE}/Mingfei/models/actor/tip/files/meshes/walk.dae",
    f"{FUEL_BASE}/OpenRobotics/models/Male visitor/2/files/meshes/MaleVisitorWalk.dae",
]

# Static standing person models
STATIC_PEOPLE = [
    {"key": "female_visitor", "uri": f"{FUEL_BASE}/OpenRobotics/models/FemaleVisitor", "z_off": 0.0},
    {"key": "walking_person", "uri": f"{FUEL_BASE}/OpenRobotics/models/Walking person", "z_off": 0.02},
]

# ---------------------------------------------------------------------------
# Density profiles for realistic scenes
# ---------------------------------------------------------------------------

REALISTIC_PROFILES = {
    "light": {"people_walking": 2, "people_standing": 2, "vehicles": 2},
    "medium": {"people_walking": 5, "people_standing": 5, "vehicles": 5},
    "heavy": {"people_walking": 12, "people_standing": 13, "vehicles": 10},
}

SPEED_RANGES = {"slow": (5.0, 8.0), "normal": (3.0, 6.0), "fast": (2.0, 4.0)}

# ---------------------------------------------------------------------------
# Placement types
# ---------------------------------------------------------------------------


@dataclass
class PersonPlacement:
    x: float
    y: float
    z: float
    yaw: float
    semantic_name: str
    placement_type: str  # "walking" | "standing"
    skin_url: str = ""
    static_uri: str = ""
    z_offset: float = 0.0
    marker_color: tuple[float, float, float] = (1.0, 1.0, 0.0)
    speed_class: str = "normal"
    height_scale: float = 1.0


@dataclass
class VehiclePlacement:
    x: float
    y: float
    z: float
    yaw: float
    semantic_name: str
    fuel_uri: str
    z_offset: float = 0.0
    marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Terrain slope calculation
# ---------------------------------------------------------------------------


def terrain_slope(terrain: TerrainInfo, x: float, y: float, radius: float = 2.0) -> float:
    """Compute terrain slope angle at (x, y) in radians."""
    z_c = terrain_z(terrain, x, y)
    z_xp = terrain_z(terrain, x + radius, y)
    z_xn = terrain_z(terrain, x - radius, y)
    z_yp = terrain_z(terrain, x, y + radius)
    z_yn = terrain_z(terrain, x, y - radius)
    dx = (z_xp - z_xn) / (2 * radius)
    dy = (z_yp - z_yn) / (2 * radius)
    return math.atan(math.sqrt(dx * dx + dy * dy))


# ---------------------------------------------------------------------------
# Placement generation
# ---------------------------------------------------------------------------


def _terrain_bounds(terrain: TerrainInfo) -> tuple[float, float, float, float]:
    return (
        terrain.offset_x - terrain.width_m / 2,
        terrain.offset_y - terrain.height_m / 2,
        terrain.offset_x + terrain.width_m / 2,
        terrain.offset_y + terrain.height_m / 2,
    )


def generate_realistic_placements(
    terrain: TerrainInfo,
    density: str,
    seed: int,
    exclusion_zones: list[tuple[float, float, float]] | None = None,
    clusters: list[tuple[float, float, float]] | None = None,
) -> tuple[list[PersonPlacement], list[VehiclePlacement]]:
    rng = random.Random(seed)
    profile = REALISTIC_PROFILES[density]

    x_min, y_min, x_max, y_max = _terrain_bounds(terrain)
    margin = 20.0
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin

    if exclusion_zones is None:
        exclusion_zones = [(0.0, 0.0, 15.0)]
    if clusters is None:
        clusters = []

    def in_exclusion(x: float, y: float) -> bool:
        return any(math.hypot(x - ex, y - ey) < er for ex, ey, er in exclusion_zones)

    def random_pos() -> tuple[float, float]:
        for _ in range(100):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if not in_exclusion(x, y):
                return x, y
        return rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)

    def flat_pos() -> tuple[float, float]:
        for _ in range(200):
            x, y = random_pos()
            if terrain_slope(terrain, x, y) < math.radians(15):
                return x, y
        return random_pos()

    # --- People ---
    people: list[PersonPlacement] = []
    n_walking = profile["people_walking"]
    n_standing = profile["people_standing"]

    for i in range(n_walking + n_standing):
        is_walking = i < n_walking
        pdef = PERSON_DEFS[i % len(PERSON_DEFS)]
        speed_class = pdef[8]
        marker_rgb = pdef[6]  # backpack color for re-ID marker

        # Place near clusters for concealment, else scattered
        if i < len(clusters):
            cx, cy, cr = clusters[i % len(clusters)]
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(1.5, cr * 0.6)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
        else:
            x, y = random_pos()

        z = terrain_z(terrain, x, y)
        yaw = rng.uniform(0, 2 * math.pi)

        if is_walking:
            skin = ACTOR_SKINS[i % len(ACTOR_SKINS)]
            name = f"person_walking_{i:03d}"
            people.append(PersonPlacement(
                x, y, z, yaw, name, "walking",
                skin_url=skin, z_offset=0.0,
                marker_color=marker_rgb, speed_class=speed_class,
                height_scale=pdef[5],
            ))
        else:
            static = STATIC_PEOPLE[i % len(STATIC_PEOPLE)]
            name = f"person_standing_{i:03d}"
            people.append(PersonPlacement(
                x, y, z, yaw, name, "standing",
                static_uri=static["uri"], z_offset=static["z_off"],
                marker_color=marker_rgb,
            ))

    # --- Vehicles ---
    vehicles: list[VehiclePlacement] = []
    n_vehicles = profile["vehicles"]

    for i in range(n_vehicles):
        vdef = VEHICLE_DEFS[i % len(VEHICLE_DEFS)]
        fuel_v = FUEL_VEHICLES[i % len(FUEL_VEHICLES)]
        marker_rgb = vdef[8]  # roof stripe color

        x, y = flat_pos()
        z = terrain_z(terrain, x, y)
        yaw = rng.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2]) + rng.uniform(-0.15, 0.15)

        name = f"vehicle_{fuel_v['key']}_{i:03d}"
        vehicles.append(VehiclePlacement(
            x, y, z, yaw, name,
            fuel_uri=fuel_v["uri"],
            z_offset=fuel_v["z_off"],
            marker_color=marker_rgb,
        ))

    return people, vehicles


# ---------------------------------------------------------------------------
# SDF generation — Fuel model includes and actors
# ---------------------------------------------------------------------------


def _add_reid_marker(world: ET.Element, x: float, y: float, z: float,
                     color: tuple[float, float, float], name: str,
                     marker_z_off: float = 2.0, size: float = 0.2) -> None:
    """Add a small colored box marker above an entity for aerial re-ID."""
    model = ET.SubElement(world, "model", name=f"marker_{name}")
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x:.2f} {y:.2f} {z + marker_z_off:.2f} 0 0 0"
    link = ET.SubElement(model, "link", name="link")
    vis = ET.SubElement(link, "visual", name="marker")
    geom = ET.SubElement(vis, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{size:.2f} {size:.2f} 0.05"
    m = ET.SubElement(vis, "material")
    r, g, b = color
    ET.SubElement(m, "ambient").text = f"{r:.2f} {g:.2f} {b:.2f} 1"
    ET.SubElement(m, "diffuse").text = f"{r:.2f} {g:.2f} {b:.2f} 1"


def _add_fuel_vehicle_include(world: ET.Element, terrain: TerrainInfo,
                              vp: VehiclePlacement) -> None:
    """Add a <include> element for a static Fuel vehicle model."""
    z = terrain_z(terrain, vp.x, vp.y) + vp.z_offset
    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "name").text = vp.semantic_name
    ET.SubElement(inc, "pose").text = f"{vp.x:.2f} {vp.y:.2f} {z:.2f} 0 0 {vp.yaw:.2f}"
    ET.SubElement(inc, "uri").text = vp.fuel_uri
    # Roof marker for re-ID
    _add_reid_marker(world, vp.x, vp.y, z, vp.marker_color,
                     vp.semantic_name, marker_z_off=2.5, size=0.35)


def _add_standing_person_include(world: ET.Element, terrain: TerrainInfo,
                                 pp: PersonPlacement) -> None:
    """Add a <include> for a static Fuel person model."""
    z = terrain_z(terrain, pp.x, pp.y) + pp.z_offset
    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "name").text = pp.semantic_name
    ET.SubElement(inc, "pose").text = f"{pp.x:.2f} {pp.y:.2f} {z:.2f} 0 0 {pp.yaw:.2f}"
    ET.SubElement(inc, "uri").text = pp.static_uri
    # Shoulder marker for re-ID
    _add_reid_marker(world, pp.x, pp.y, z, pp.marker_color,
                     pp.semantic_name, marker_z_off=1.8, size=0.15)


def _add_walking_actor_realistic(
    world: ET.Element,
    terrain: TerrainInfo,
    pp: PersonPlacement,
    rng: random.Random,
) -> None:
    """Add a walking <actor> with terrain-aware waypoint Z."""
    actor = ET.SubElement(world, "actor", name=pp.semantic_name)

    skin = ET.SubElement(actor, "skin")
    ET.SubElement(skin, "filename").text = pp.skin_url
    ET.SubElement(skin, "scale").text = f"{pp.height_scale:.2f}"

    anim = ET.SubElement(actor, "animation", name="walk")
    ET.SubElement(anim, "filename").text = pp.skin_url
    ET.SubElement(anim, "interpolate_x").text = "true"

    script = ET.SubElement(actor, "script")
    ET.SubElement(script, "loop").text = "true"
    ET.SubElement(script, "delay_start").text = f"{rng.uniform(0, 3):.1f}"
    ET.SubElement(script, "auto_start").text = "true"

    traj = ET.SubElement(script, "trajectory", id="0", type="walk", tension="0.5")

    radius = rng.uniform(2.0, 5.0)
    points = rng.randint(4, 6)
    speed_lo, speed_hi = SPEED_RANGES.get(pp.speed_class, (3.0, 6.0))

    for i in range(points + 1):
        t = i / points * 2 * math.pi
        wx = pp.x + radius * math.cos(t)
        wy = pp.y + radius * math.sin(t)
        wz = terrain_z(terrain, wx, wy) + pp.z_offset + 1.0
        time_s = i * rng.uniform(speed_lo, speed_hi)
        yaw = math.atan2(wy - pp.y, wx - pp.x)

        wp = ET.SubElement(traj, "waypoint")
        ET.SubElement(wp, "time").text = f"{time_s:.1f}"
        ET.SubElement(wp, "pose").text = f"{wx:.2f} {wy:.2f} {wz:.2f} 0 0 {yaw:.2f}"

    # Static color marker at home position
    _add_reid_marker(world, pp.x, pp.y, terrain_z(terrain, pp.x, pp.y),
                     pp.marker_color, pp.semantic_name, marker_z_off=1.8, size=0.15)


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


def generate_realistic_world_sdf(
    terrain_name: str,
    density: str,
    seed: int,
    exclusion_zones: list[tuple[float, float, float]] | None = None,
) -> tuple[str, list[tuple[float, float, float]]]:
    """Generate complete SDF with vegetation (inline) + realistic Fuel people/vehicles."""
    terrain = load_terrain(terrain_name)
    profile = DENSITY_PROFILES[density]

    world_name = f"{terrain_name}_realistic_{density}"
    sdf, world = _sdf_header(world_name, profile["shadows"])

    # Terrain + helipad
    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "uri").text = f"model://{terrain_name}"
    inc2 = ET.SubElement(world, "include")
    ET.SubElement(inc2, "uri").text = "model://helipad"

    # Generate vegetation using existing system
    veg_placements, _, _, clusters = generate_placements(
        terrain, density, seed, exclusion_zones,
    )

    # Generate realistic people + vehicles
    people, vehicles = generate_realistic_placements(
        terrain, density, seed, exclusion_zones, clusters,
    )

    # Add vegetation
    for idx, p in enumerate(veg_placements):
        _add_inline_vegetation(world, idx, p)

    # Add people
    for i, pp in enumerate(people):
        if pp.placement_type == "walking":
            _add_walking_actor_realistic(world, terrain, pp, random.Random(seed + i))
        else:
            _add_standing_person_include(world, terrain, pp)

    # Add vehicles
    for vp in vehicles:
        _add_fuel_vehicle_include(world, terrain, vp)

    # Add concealment markers
    _add_concealment_markers(world, clusters, terrain)

    ET.indent(sdf, space="  ")
    xml_str = '<?xml version="1.0" ?>\n' + ET.tostring(sdf, encoding="unicode")
    return xml_str, clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place realistic Fuel mesh targets on real-terrain world",
    )
    parser.add_argument("--world", required=True, help="World/model name (e.g. 1779343687303)")
    parser.add_argument("--density", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Generate all three densities")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Verify Fuel assets
    registry = verify_all()
    missing = [k for k, v in registry.items() if not v.cache_dir or not v.cache_dir.exists()]
    if missing:
        print(f"ERROR: {len(missing)} Fuel models not cached. Run: python -m apps.tools.import_fuel_assets")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    densities = ["light", "medium", "heavy"] if args.all else [args.density]
    out_dir = Path(args.output_dir) if args.output_dir else GZ_WORLDS_DIR

    for density in densities:
        print(f"\nGenerating {args.world}_realistic_{density}.sdf ...")
        xml_str, clusters = generate_realistic_world_sdf(args.world, density, args.seed)

        out_path = out_dir / f"{args.world}_realistic_{density}.sdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml_str)

        n_veg = xml_str.count('<model name="veg_')
        n_actors = xml_str.count("<actor")
        n_includes = xml_str.count("<include>") - 2  # subtract terrain + helipad
        n_vehicles = xml_str.count('<name>vehicle_')
        n_markers = xml_str.count('<model name="marker_')

        print(f"  Written: {out_path}")
        print(f"  Vegetation: {n_veg}")
        print(f"  People (actors + static): {n_actors + (n_includes - n_vehicles)}")
        print(f"    Walking actors: {n_actors}")
        print(f"  Vehicles: {n_vehicles}")
        print(f"  Re-ID markers: {n_markers}")
        print(f"  Concealment clusters: {len(clusters)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
