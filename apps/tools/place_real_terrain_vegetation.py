"""Place lightweight vegetation, distinct people, and vehicles on real-terrain Gazebo worlds.

Reads the heightmap TIF to compute ground-level Z for each entity and generates
overlay SDF files with light/medium/heavy density. All geometry is inlined as
static SDF — no external model dependencies except Fuel actors for walking people.

People have unique color combinations for re-identification testing.
Vehicles have distinct colors and types.

Usage:
    python -m apps.tools.place_real_terrain_vegetation --world 1779343687303 --density medium
    python -m apps.tools.place_real_terrain_vegetation --world 1779343687303 --all
    python -m apps.tools.place_real_terrain_vegetation --world 1779343687303 --audit
"""

from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PX4_DIR = PROJECT_ROOT.parent / "PX4-Autopilot"
GZ_MODELS_DIR = PX4_DIR / "Tools" / "simulation" / "gz" / "models"
GZ_WORLDS_DIR = PX4_DIR / "Tools" / "simulation" / "gz" / "worlds"

# ---------------------------------------------------------------------------
# Terrain height from heightmap
# ---------------------------------------------------------------------------


class TerrainInfo(NamedTuple):
    width_m: float
    height_m: float
    elevation_m: float
    offset_x: float
    offset_y: float
    offset_z: float
    pixels: list[list[float]]
    img_w: int
    img_h: int


def _load_heightmap_px(heightmap_path: Path) -> list[list[float]]:
    from PIL import Image

    img = Image.open(heightmap_path)
    img = img.convert("L")
    w, h = img.size
    raw = list(img.getdata())
    return [[raw[y * w + x] / 255.0 for x in range(w)] for y in range(h)]


def load_terrain(world_name: str) -> TerrainInfo:
    model_sdf = GZ_MODELS_DIR / world_name / "model.sdf"
    heightmap_path = GZ_MODELS_DIR / world_name / "textures" / f"{world_name}_height_map.tif"

    if not heightmap_path.exists():
        print(f"WARNING: heightmap not found at {heightmap_path}, using Z=0 for all vegetation")
        return TerrainInfo(0, 0, 0, 0, 0, 0, [], 0, 0)

    tree = ET.parse(model_sdf)
    root = tree.getroot()

    pose_el = root.find(".//model/pose")
    if pose_el is not None and pose_el.text:
        parts = [float(v) for v in pose_el.text.strip().split()]
        off_x, off_y, off_z = parts[0], parts[1], parts[2]
    else:
        off_x, off_y, off_z = 0.0, 0.0, 0.0

    hm_size_el = root.find(".//heightmap/size")
    if hm_size_el is not None and hm_size_el.text:
        parts = [float(v) for v in hm_size_el.text.strip().split()]
        w_m, h_m, elev_m = parts[0], parts[1], parts[2]
    else:
        w_m, h_m, elev_m = 739.0, 736.0, 30.0

    pixels = _load_heightmap_px(heightmap_path)
    img_h = len(pixels)
    img_w = len(pixels[0]) if pixels else 0

    return TerrainInfo(w_m, h_m, elev_m, off_x, off_y, off_z, pixels, img_w, img_h)


def terrain_z(terrain: TerrainInfo, world_x: float, world_y: float) -> float:
    """Get terrain elevation Z at world (x, y) coordinates."""
    if not terrain.pixels:
        return terrain.offset_z

    local_x = world_x - terrain.offset_x + terrain.width_m / 2
    local_y = world_y - terrain.offset_y + terrain.height_m / 2

    px = int(local_x / terrain.width_m * (terrain.img_w - 1))
    py = int(local_y / terrain.height_m * (terrain.img_h - 1))

    px = max(0, min(px, terrain.img_w - 1))
    py = max(0, min(py, terrain.img_h - 1))

    normalized = terrain.pixels[py][px]
    return terrain.offset_z + normalized * terrain.elevation_m


# ---------------------------------------------------------------------------
# Improved vegetation geometry — multi-primitive for better silhouette
# ---------------------------------------------------------------------------

VEG_GEOMETRY = {
    "tree": {
        "trunk_r": 0.15, "trunk_h": 3.0,
        "canopy_r": 2.0, "canopy_type": "sphere",
        "canopy_z": 4.2,
        "trunk_rgb": (0.45, 0.32, 0.18),
        "canopy_rgb": (0.20, 0.50, 0.15),
        # Extra: second canopy lobe for fuller tree
        "extra_canopies": [
            {"r": 1.4, "z_off": 1.2, "x_off": 1.0, "y_off": 0.5},
            {"r": 1.2, "z_off": 0.8, "x_off": -0.8, "y_off": -0.6},
        ],
    },
    "bush": {
        "trunk_r": 0.0, "trunk_h": 0.0,
        "canopy_r": 0.8, "canopy_type": "sphere",
        "canopy_z": 0.7,
        "trunk_rgb": (0, 0, 0),
        "canopy_rgb": (0.22, 0.52, 0.18),
        "extra_canopies": [
            {"r": 0.55, "z_off": -0.1, "x_off": 0.5, "y_off": 0.3},
            {"r": 0.5, "z_off": 0.05, "x_off": -0.4, "y_off": -0.4},
        ],
    },
    "shrub": {
        "trunk_r": 0.0, "trunk_h": 0.0,
        "canopy_r": 0.5, "canopy_type": "cylinder",
        "canopy_z": 0.4, "canopy_h": 0.8,
        "trunk_rgb": (0, 0, 0),
        "canopy_rgb": (0.25, 0.48, 0.20),
        "extra_canopies": [],
    },
    "palm": {
        "trunk_r": 0.12, "trunk_h": 5.0,
        "canopy_r": 2.5, "canopy_type": "cylinder",
        "canopy_z": 5.3, "canopy_h": 0.6,
        "trunk_rgb": (0.55, 0.45, 0.30),
        "canopy_rgb": (0.18, 0.52, 0.14),
        "extra_canopies": [],
    },
    "grass": {
        "trunk_r": 0.0, "trunk_h": 0.0,
        "canopy_r": 1.5, "canopy_type": "cylinder",
        "canopy_z": 0.15, "canopy_h": 0.3,
        "trunk_rgb": (0, 0, 0),
        "canopy_rgb": (0.30, 0.55, 0.20),
        "extra_canopies": [],
    },
    "rock": {
        "trunk_r": 0.0, "trunk_h": 0.0,
        "canopy_r": 0.6, "canopy_type": "sphere",
        "canopy_z": 0.35,
        "trunk_rgb": (0, 0, 0),
        "canopy_rgb": (0.55, 0.52, 0.48),
        "extra_canopies": [],
    },
}

SCALE_RANGES = {
    "tree": (0.8, 1.4),
    "bush": (0.6, 1.2),
    "shrub": (0.5, 1.0),
    "palm": (0.7, 1.3),
    "grass": (0.8, 1.5),
    "rock": (0.4, 1.2),
}

# ---------------------------------------------------------------------------
# Distinct people definitions — unique color combinations
# ---------------------------------------------------------------------------

# Each person: (semantic_name, shirt_rgb, pants_rgb, skin_rgb, hat_rgb_or_None,
#                height_scale, backpack_rgb, width_scale, speed_class)
# speed_class: "slow" (5-8s/waypoint), "normal" (3-6s), "fast" (2-4s)
# backpack_rgb: small colored box visible from aerial overhead — unique per person
PERSON_DEFS = [
    ("person_red_shirt",     (0.80, 0.15, 0.10), (0.15, 0.15, 0.50), (0.70, 0.55, 0.40), None,                  1.00, (0.90, 0.70, 0.10), 1.00, "normal"),
    ("person_blue_jacket",   (0.10, 0.20, 0.80), (0.35, 0.32, 0.28), (0.65, 0.50, 0.35), None,                  0.95, (0.20, 0.80, 0.20), 0.95, "normal"),
    ("person_green_vest",    (0.15, 0.70, 0.20), (0.30, 0.25, 0.15), (0.60, 0.45, 0.30), (0.60, 0.45, 0.30),   1.05, (0.80, 0.10, 0.10), 1.00, "fast"),
    ("person_white_hat",     (0.90, 0.90, 0.90), (0.20, 0.20, 0.60), (0.70, 0.55, 0.40), (0.95, 0.95, 0.95),   1.00, (0.60, 0.10, 0.60), 1.05, "slow"),
    ("person_orange_pack",   (0.90, 0.50, 0.10), (0.22, 0.22, 0.22), (0.55, 0.40, 0.25), None,                  0.90, (0.95, 0.55, 0.15), 0.90, "normal"),
    ("person_yellow_rain",   (0.85, 0.85, 0.15), (0.28, 0.28, 0.28), (0.65, 0.50, 0.35), (0.85, 0.85, 0.15),   0.98, (0.10, 0.50, 0.80), 1.00, "fast"),
    ("person_gray_coat",     (0.50, 0.50, 0.55), (0.15, 0.15, 0.45), (0.60, 0.45, 0.30), None,                  1.10, (0.80, 0.20, 0.20), 1.05, "slow"),
    ("person_purple_top",    (0.55, 0.15, 0.65), (0.18, 0.18, 0.18), (0.55, 0.40, 0.25), None,                  0.92, (0.90, 0.80, 0.10), 0.92, "normal"),
    ("person_khaki_shorts",  (0.75, 0.70, 0.45), (0.55, 0.50, 0.30), (0.70, 0.55, 0.40), (0.40, 0.35, 0.25),   1.00, (0.10, 0.60, 0.60), 1.00, "normal"),
    ("person_black_hoodie",  (0.10, 0.10, 0.10), (0.12, 0.12, 0.35), (0.65, 0.50, 0.35), None,                  1.05, (0.85, 0.85, 0.10), 0.95, "fast"),
    ("person_camo_pants",    (0.30, 0.35, 0.25), (0.35, 0.40, 0.20), (0.60, 0.45, 0.30), (0.25, 0.30, 0.20),   0.97, (0.70, 0.50, 0.20), 1.00, "normal"),
    ("person_pink_scarf",    (0.85, 0.40, 0.55), (0.15, 0.15, 0.40), (0.70, 0.55, 0.40), None,                  0.93, (0.20, 0.20, 0.80), 0.90, "slow"),
    ("person_brown_jacket",  (0.55, 0.35, 0.20), (0.32, 0.30, 0.28), (0.65, 0.50, 0.35), (0.55, 0.35, 0.20),   1.02, (0.15, 0.70, 0.15), 1.05, "normal"),
    ("person_teal_shirt",    (0.10, 0.65, 0.60), (0.20, 0.20, 0.20), (0.60, 0.45, 0.30), None,                  0.88, (0.80, 0.10, 0.80), 0.88, "fast"),
    ("person_red_cap",       (0.85, 0.85, 0.85), (0.20, 0.45, 0.20), (0.55, 0.40, 0.25), (0.80, 0.15, 0.10),   1.08, (0.50, 0.50, 0.50), 1.08, "normal"),
    ("person_maroon_sweat",  (0.50, 0.10, 0.15), (0.40, 0.38, 0.35), (0.70, 0.55, 0.40), None,                  0.95, (0.10, 0.70, 0.70), 0.95, "slow"),
    ("person_navy_suit",     (0.10, 0.10, 0.35), (0.10, 0.10, 0.35), (0.65, 0.50, 0.35), None,                  1.00, (0.80, 0.60, 0.10), 1.02, "normal"),
    ("person_lime_tank",     (0.55, 0.90, 0.15), (0.20, 0.20, 0.55), (0.55, 0.40, 0.25), None,                  0.90, (0.70, 0.10, 0.10), 0.92, "fast"),
    ("person_denim_jacket",  (0.25, 0.35, 0.55), (0.25, 0.35, 0.55), (0.60, 0.45, 0.30), None,                  1.03, (0.10, 0.80, 0.40), 1.00, "normal"),
    ("person_beige_vest",    (0.80, 0.75, 0.60), (0.35, 0.30, 0.20), (0.70, 0.55, 0.40), (0.80, 0.75, 0.60),   0.96, (0.60, 0.20, 0.70), 0.95, "slow"),
]

SPEED_RANGES = {
    "slow": (5.0, 8.0),
    "normal": (3.0, 6.0),
    "fast": (2.0, 4.0),
}

# ---------------------------------------------------------------------------
# Vehicle definitions — distinct types and colors
# ---------------------------------------------------------------------------

# Each: (semantic_name, body_rgb, roof_rgb, length, width, height, wheel_r, wheelbase_offset, stripe_rgb)
VEHICLE_DEFS = [
    ("sedan_red",      (0.80, 0.12, 0.10), (0.70, 0.10, 0.08), 4.5, 1.8, 1.4, 0.35, 1.3, (0.95, 0.95, 0.10)),
    ("sedan_blue",     (0.10, 0.20, 0.75), (0.08, 0.15, 0.65), 4.5, 1.8, 1.4, 0.35, 1.3, (0.90, 0.90, 0.90)),
    ("sedan_white",    (0.92, 0.92, 0.92), (0.85, 0.85, 0.85), 4.5, 1.8, 1.4, 0.35, 1.3, (0.10, 0.10, 0.70)),
    ("pickup_black",   (0.12, 0.12, 0.12), (0.10, 0.10, 0.10), 5.5, 2.0, 1.8, 0.42, 1.8, (0.80, 0.15, 0.10)),
    ("pickup_green",   (0.15, 0.45, 0.15), (0.12, 0.38, 0.12), 5.5, 2.0, 1.8, 0.42, 1.8, (0.90, 0.80, 0.10)),
    ("van_white",      (0.90, 0.90, 0.90), (0.85, 0.85, 0.85), 5.0, 2.0, 2.2, 0.38, 1.5, (0.15, 0.60, 0.80)),
    ("van_yellow",     (0.90, 0.80, 0.10), (0.85, 0.75, 0.08), 5.0, 2.0, 2.2, 0.38, 1.5, (0.10, 0.10, 0.10)),
    ("suv_silver",     (0.65, 0.65, 0.68), (0.58, 0.58, 0.60), 5.0, 2.0, 1.8, 0.40, 1.5, (0.80, 0.10, 0.70)),
    ("truck_brown",    (0.55, 0.35, 0.20), (0.45, 0.28, 0.15), 7.0, 2.4, 2.5, 0.50, 2.5, (0.15, 0.70, 0.15)),
    ("suv_red",        (0.75, 0.12, 0.10), (0.65, 0.10, 0.08), 5.0, 2.0, 1.8, 0.40, 1.5, (0.10, 0.10, 0.80)),
]

# ---------------------------------------------------------------------------
# Density profiles
# ---------------------------------------------------------------------------

DENSITY_PROFILES = {
    "light": {
        "mix": {"tree": 5, "bush": 5, "shrub": 4, "palm": 2, "grass": 2, "rock": 2},
        "clusters": 2,
        "cluster_size": (3, 6),
        "people": 4,
        "vehicles": 3,
        "shadows": False,
    },
    "medium": {
        "mix": {"tree": 18, "bush": 18, "shrub": 14, "palm": 8, "grass": 10, "rock": 7},
        "clusters": 4,
        "cluster_size": (4, 9),
        "people": 8,
        "vehicles": 5,
        "shadows": True,
    },
    "heavy": {
        "mix": {"tree": 45, "bush": 45, "shrub": 35, "palm": 20, "grass": 20, "rock": 15},
        "clusters": 8,
        "cluster_size": (5, 14),
        "people": 15,
        "vehicles": 8,
        "shadows": True,
    },
    "camera_lite": {
        "mix": {"tree": 2, "bush": 2, "shrub": 2, "palm": 1, "grass": 1, "rock": 1},
        "clusters": 1,
        "cluster_size": (2, 4),
        "people": 2,
        "vehicles": 1,
        "shadows": False,
    },
    "realistic_lite": {
        "mix": {"tree": 4, "bush": 6, "shrub": 4, "palm": 0, "grass": 2, "rock": 2},
        "clusters": 2,
        "cluster_size": (2, 5),
        "people": 10,
        "vehicles": 3,
        "shadows": False,
    },
}

# ---------------------------------------------------------------------------
# Placement types
# ---------------------------------------------------------------------------


class Placement(NamedTuple):
    x: float
    y: float
    z: float
    yaw: float
    scale: float
    model_type: str
    semantic_name: str = ""
    # Person-specific
    shirt_rgb: tuple[float, float, float] | None = None
    pants_rgb: tuple[float, float, float] | None = None
    hat_rgb: tuple[float, float, float] | None = None
    height_scale: float = 1.0
    backpack_rgb: tuple[float, float, float] | None = None
    width_scale: float = 1.0
    speed_class: str = "normal"
    # Vehicle-specific
    body_rgb: tuple[float, float, float] | None = None
    roof_rgb: tuple[float, float, float] | None = None
    veh_length: float = 0.0
    veh_width: float = 0.0
    veh_height: float = 0.0
    stripe_rgb: tuple[float, float, float] | None = None


def _terrain_bounds(terrain: TerrainInfo) -> tuple[float, float, float, float]:
    x_min = terrain.offset_x - terrain.width_m / 2
    x_max = terrain.offset_x + terrain.width_m / 2
    y_min = terrain.offset_y - terrain.height_m / 2
    y_max = terrain.offset_y + terrain.height_m / 2
    return x_min, y_min, x_max, y_max


def _generate_clusters(
    rng: random.Random,
    n_clusters: int,
    x_min: float, y_min: float, x_max: float, y_max: float,
    exclusion_zones: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    clusters = []
    for _ in range(n_clusters):
        for _ in range(50):
            cx = rng.uniform(x_min, x_max)
            cy = rng.uniform(y_min, y_max)
            if not any(math.hypot(cx - ex, cy - ey) < er for ex, ey, er in exclusion_zones):
                clusters.append((cx, cy, rng.uniform(5.0, 12.0)))
                break
    return clusters


def generate_placements(
    terrain: TerrainInfo,
    density: str,
    seed: int,
    exclusion_zones: list[tuple[float, float, float]] | None = None,
) -> tuple[list[Placement], list[Placement], list[Placement], list[tuple[float, float, float]]]:
    """Returns (vegetation, people, vehicles, clusters)."""
    rng = random.Random(seed)
    profile = DENSITY_PROFILES[density]

    if density == "realistic_lite":
        x_min, x_max = 170.0, 270.0
        y_min, y_max = -400.0, -300.0
    else:
        x_min, y_min, x_max, y_max = _terrain_bounds(terrain)
        margin = 20.0
        x_min += margin
        x_max -= margin
        y_min += margin
        y_max -= margin

    if exclusion_zones is None:
        exclusion_zones = [(0.0, 0.0, 15.0)]

    def in_exclusion(x: float, y: float) -> bool:
        return any(math.hypot(x - ex, y - ey) < er for ex, ey, er in exclusion_zones)

    def random_pos() -> tuple[float, float]:
        for _ in range(100):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if not in_exclusion(x, y):
                return x, y
        return rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)

    clusters = _generate_clusters(rng, profile["clusters"], x_min, y_min, x_max, y_max, exclusion_zones)

    # --- Vegetation ---
    placements: list[Placement] = []
    for model_type, count in profile["mix"].items():
        for _ in range(count):
            x, y = random_pos()
            z = terrain_z(terrain, x, y)
            yaw = rng.uniform(0, 2 * math.pi)
            scale = rng.uniform(*SCALE_RANGES[model_type])
            placements.append(Placement(x, y, z, yaw, scale, model_type))

    cluster_lo, cluster_hi = profile["cluster_size"]
    for cx, cy, cr in clusters:
        n = rng.randint(cluster_lo, cluster_hi)
        for _ in range(n):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(1.0, cr)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            if in_exclusion(x, y):
                continue
            z = terrain_z(terrain, x, y)
            model_type = rng.choice(["bush", "shrub", "tree", "bush", "grass"])
            yaw = rng.uniform(0, 2 * math.pi)
            scale = rng.uniform(*SCALE_RANGES[model_type])
            placements.append(Placement(x, y, z, yaw, scale, model_type))

    # --- People near clusters + scattered ---
    n_people = min(profile["people"], len(PERSON_DEFS))
    people: list[Placement] = []
    for i in range(n_people):
        pdef = PERSON_DEFS[i % len(PERSON_DEFS)]

        # Some people near clusters, some scattered
        if i < len(clusters):
            cx, cy, cr = clusters[i % len(clusters)]
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(1.5, cr * 0.6)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
        else:
            x, y = random_pos()

        z = terrain_z(terrain, x, y)
        people.append(Placement(
            x, y, z, rng.uniform(0, 2 * math.pi), 1.0, "person",
            semantic_name=pdef[0], shirt_rgb=pdef[1], pants_rgb=pdef[2],
            hat_rgb=pdef[4], height_scale=pdef[5],
            backpack_rgb=pdef[6], width_scale=pdef[7], speed_class=pdef[8],
        ))

    # --- Vehicles ---
    n_vehicles = min(profile["vehicles"], len(VEHICLE_DEFS))
    vehicles: list[Placement] = []
    for i in range(n_vehicles):
        vdef = VEHICLE_DEFS[i % len(VEHICLE_DEFS)]
        x, y = random_pos()
        z = terrain_z(terrain, x, y)
        yaw = rng.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2]) + rng.uniform(-0.15, 0.15)
        vehicles.append(Placement(
            x, y, z, yaw, 1.0, "vehicle",
            semantic_name=vdef[0], body_rgb=vdef[1], roof_rgb=vdef[2],
            veh_length=vdef[3], veh_width=vdef[4], veh_height=vdef[5],
            stripe_rgb=vdef[8],
        ))

    return placements, people, vehicles, clusters


# ---------------------------------------------------------------------------
# SDF generation
# ---------------------------------------------------------------------------

def _mat(el: ET.Element, r: float, g: float, b: float, a: float = 1.0) -> None:
    m = ET.SubElement(el, "material")
    ET.SubElement(m, "ambient").text = f"{r:.2f} {g:.2f} {b:.2f} {a:.2f}"
    ET.SubElement(m, "diffuse").text = f"{min(r + 0.05, 1):.2f} {min(g + 0.05, 1):.2f} {min(b + 0.05, 1):.2f} {a:.2f}"


def _add_inline_vegetation(world: ET.Element, idx: int, p: Placement) -> None:
    geo = VEG_GEOMETRY[p.model_type]
    s = p.scale

    model = ET.SubElement(world, "model", name=f"veg_{p.model_type}_{idx}")
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{p.x:.2f} {p.y:.2f} {p.z:.2f} 0 0 {p.yaw:.2f}"
    link = ET.SubElement(model, "link", name="link")

    vis_count = 0

    # Trunk
    if geo["trunk_h"] > 0:
        vis_count += 1
        trunk = ET.SubElement(link, "visual", name=f"trunk_{vis_count}")
        geom = ET.SubElement(trunk, "geometry")
        cyl = ET.SubElement(geom, "cylinder")
        ET.SubElement(cyl, "radius").text = f"{geo['trunk_r'] * s:.3f}"
        ET.SubElement(cyl, "length").text = f"{geo['trunk_h'] * s:.2f}"
        ET.SubElement(trunk, "pose").text = f"0 0 {geo['trunk_h'] * s / 2:.2f} 0 0 0"
        tr, tg, tb = geo["trunk_rgb"]
        _mat(trunk, tr, tg, tb)

    # Main canopy
    vis_count += 1
    canopy = ET.SubElement(link, "visual", name=f"canopy_{vis_count}")
    geom = ET.SubElement(canopy, "geometry")
    cr, cg, cb = geo["canopy_rgb"]
    rng_local = random.Random(idx)
    cr = max(0, min(1, cr + rng_local.uniform(-0.04, 0.04)))
    cg = max(0, min(1, cg + rng_local.uniform(-0.04, 0.04)))
    cb = max(0, min(1, cb + rng_local.uniform(-0.03, 0.03)))

    if geo["canopy_type"] == "sphere":
        sph = ET.SubElement(geom, "sphere")
        ET.SubElement(sph, "radius").text = f"{geo['canopy_r'] * s:.2f}"
    else:
        cyl = ET.SubElement(geom, "cylinder")
        ET.SubElement(cyl, "radius").text = f"{geo['canopy_r'] * s:.2f}"
        ET.SubElement(cyl, "length").text = f"{geo.get('canopy_h', 0.5) * s:.2f}"
    ET.SubElement(canopy, "pose").text = f"0 0 {geo['canopy_z'] * s:.2f} 0 0 0"
    _mat(canopy, cr, cg, cb)

    # Extra canopy lobes for fuller appearance
    for ei, ec in enumerate(geo.get("extra_canopies", [])):
        vis_count += 1
        extra = ET.SubElement(link, "visual", name=f"extra_canopy_{vis_count}")
        geom = ET.SubElement(extra, "geometry")
        sph = ET.SubElement(geom, "sphere")
        ET.SubElement(sph, "radius").text = f"{ec['r'] * s:.2f}"
        ecr = max(0, min(1, cr + rng_local.uniform(-0.03, 0.03)))
        ecg = max(0, min(1, cg + rng_local.uniform(-0.03, 0.03)))
        ecb = max(0, min(1, cb + rng_local.uniform(-0.02, 0.02)))
        _mat(extra, ecr, ecg, ecb)
        ET.SubElement(extra, "pose").text = (
            f"{ec['x_off'] * s:.2f} {ec['y_off'] * s:.2f} "
            f"{(geo['canopy_z'] + ec['z_off']) * s:.2f} 0 0 0"
        )


def _add_person_inline(world: ET.Element, p: Placement, person_id: int) -> None:
    """Add a visually distinct inline person model.

    Uses simple geometry (boxes for torso/legs, sphere for head) with the
    unique color combination from the person definition. Includes backpack
    marker visible from aerial overhead and width variation.
    """
    name = p.semantic_name or f"person_{person_id}"
    hs = p.height_scale
    ws = p.width_scale
    z = p.z  # terrain height — feet on ground

    model = ET.SubElement(world, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{p.x:.2f} {p.y:.2f} {z:.2f} 0 0 {p.yaw:.2f}"
    link = ET.SubElement(model, "link", name="link")

    sr, sg, sb = p.shirt_rgb or (0.5, 0.5, 0.5)
    pr, pg, pb = p.pants_rgb or (0.3, 0.3, 0.3)

    # Head
    head = ET.SubElement(link, "visual", name="head")
    geom = ET.SubElement(head, "geometry")
    sph = ET.SubElement(geom, "sphere")
    ET.SubElement(sph, "radius").text = f"{0.14 * hs:.3f}"
    ET.SubElement(head, "pose").text = f"0 0 {1.65 * hs:.3f} 0 0 0"
    _mat(head, 0.65, 0.50, 0.35)

    # Hat (if defined)
    if p.hat_rgb is not None:
        hat = ET.SubElement(link, "visual", name="hat")
        hat_geom = ET.SubElement(hat, "geometry")
        cyl = ET.SubElement(hat_geom, "cylinder")
        ET.SubElement(cyl, "radius").text = f"{0.18 * hs:.3f}"
        ET.SubElement(cyl, "length").text = f"{0.08 * hs:.3f}"
        ET.SubElement(hat, "pose").text = f"0 0 {1.78 * hs:.3f} 0 0 0"
        hr, hg, hb = p.hat_rgb
        _mat(hat, hr, hg, hb)

    # Torso / shirt (width affected by width_scale)
    torso = ET.SubElement(link, "visual", name="torso")
    geom = ET.SubElement(torso, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{0.35 * hs * ws:.3f} {0.22 * hs * ws:.3f} {0.50 * hs:.3f}"
    ET.SubElement(torso, "pose").text = f"0 0 {1.20 * hs:.3f} 0 0 0"
    _mat(torso, sr, sg, sb)

    # Backpack / shoulder marker (visible from overhead for aerial Re-ID)
    if p.backpack_rgb is not None:
        bp = ET.SubElement(link, "visual", name="backpack")
        bp_geom = ET.SubElement(bp, "geometry")
        bx = ET.SubElement(bp_geom, "box")
        ET.SubElement(bx, "size").text = f"{0.15 * hs:.3f} {0.18 * hs * ws:.3f} {0.15 * hs:.3f}"
        ET.SubElement(bp, "pose").text = f"0 {-0.14 * hs * ws:.3f} {1.15 * hs:.3f} 0 0 0"
        br, bg, bb = p.backpack_rgb
        _mat(bp, br, bg, bb)

    # Upper legs / pants (width affected by width_scale)
    legs = ET.SubElement(link, "visual", name="legs")
    geom = ET.SubElement(legs, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{0.34 * hs * ws:.3f} {0.20 * hs * ws:.3f} {0.50 * hs:.3f}"
    ET.SubElement(legs, "pose").text = f"0 0 {0.70 * hs:.3f} 0 0 0"
    _mat(legs, pr, pg, pb)

    # Lower legs (slightly different shade for depth)
    lower = ET.SubElement(link, "visual", name="lower_legs")
    geom = ET.SubElement(lower, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{0.28 * hs * ws:.3f} {0.16 * hs * ws:.3f} {0.40 * hs:.3f}"
    ET.SubElement(lower, "pose").text = f"0 0 {0.25 * hs:.3f} 0 0 0"
    _mat(lower, pr * 0.8, pg * 0.8, pb * 0.8)


def _add_person_actor(
    world: ET.Element,
    p: Placement,
    actor_id: int,
    rng: random.Random,
) -> None:
    """Add a walking actor with colored inline model overlay."""
    # Walking actor skeleton
    actor_name = p.semantic_name or f"person_veg_{actor_id}"
    actor = ET.SubElement(world, "actor", name=f"actor_{actor_name}")

    skin = ET.SubElement(actor, "skin")
    ET.SubElement(skin, "filename").text = (
        "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"
    )
    ET.SubElement(skin, "scale").text = f"{p.height_scale:.2f}"

    anim = ET.SubElement(actor, "animation", name="walk")
    ET.SubElement(anim, "filename").text = (
        "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"
    )
    ET.SubElement(anim, "interpolate_x").text = "true"

    script = ET.SubElement(actor, "script")
    ET.SubElement(script, "loop").text = "true"
    ET.SubElement(script, "delay_start").text = f"{rng.uniform(0, 3):.1f}"
    ET.SubElement(script, "auto_start").text = "true"

    traj = ET.SubElement(script, "trajectory", id="0", type="walk", tension="0.5")

    radius = rng.uniform(2.0, 5.0)
    points = rng.randint(4, 6)
    speed_lo, speed_hi = SPEED_RANGES.get(p.speed_class, (3.0, 6.0))
    for i in range(points + 1):
        t = i / points * 2 * math.pi
        wx = p.x + radius * math.cos(t)
        wy = p.y + radius * math.sin(t)
        # Actor Z = terrain_z at waypoint + standing height offset (1.0)
        wz = p.z + 1.0
        time_s = i * rng.uniform(speed_lo, speed_hi)
        yaw = math.atan2(wy - p.y, wx - p.x)

        wp = ET.SubElement(traj, "waypoint")
        ET.SubElement(wp, "time").text = f"{time_s:.1f}"
        ET.SubElement(wp, "pose").text = f"{wx:.2f} {wy:.2f} {wz:.2f} 0 0 {yaw:.2f}"

    # Also place a static colored marker at the actor's home position
    # so YOLO has color information even from aerial view
    _add_person_inline(world, p._replace(z=p.z, yaw=0.0), person_id=actor_id + 1000)


def _add_vehicle_inline(world: ET.Element, p: Placement, veh_id: int) -> None:
    """Add inline vehicle model with body + roof + wheels."""
    name = p.semantic_name or f"vehicle_{veh_id}"
    z = p.z

    model = ET.SubElement(world, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{p.x:.2f} {p.y:.2f} {z:.2f} 0 0 {p.yaw:.2f}"
    link = ET.SubElement(model, "link", name="link")

    br, bg, bb = p.body_rgb or (0.5, 0.5, 0.5)
    rr, rg, rb = p.roof_rgb or (0.4, 0.4, 0.4)
    l, w, h = p.veh_length, p.veh_width, p.veh_height

    # Body (main box)
    body = ET.SubElement(link, "visual", name="body")
    geom = ET.SubElement(body, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{l:.2f} {w:.2f} {h * 0.6:.2f}"
    ET.SubElement(body, "pose").text = f"0 0 {h * 0.35:.2f} 0 0 0"
    _mat(body, br, bg, bb)

    # Cabin / roof
    cabin = ET.SubElement(link, "visual", name="cabin")
    geom = ET.SubElement(cabin, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{l * 0.5:.2f} {w * 0.9:.2f} {h * 0.45:.2f}"
    ET.SubElement(cabin, "pose").text = f"{-l * 0.05:.2f} 0 {h * 0.85:.2f} 0 0 0"
    _mat(cabin, rr, rg, rb)

    # Roof stripe (unique color marker visible from aerial for Re-ID)
    if p.stripe_rgb is not None:
        stripe = ET.SubElement(link, "visual", name="roof_stripe")
        s_geom = ET.SubElement(stripe, "geometry")
        bx = ET.SubElement(s_geom, "box")
        ET.SubElement(bx, "size").text = f"{l * 0.35:.2f} {0.15:.2f} {0.02:.2f}"
        ET.SubElement(stripe, "pose").text = f"{-l * 0.05:.2f} 0 {h * 1.08:.2f} 0 0 0"
        sr2, sg2, sb2 = p.stripe_rgb
        _mat(stripe, sr2, sg2, sb2)

    # Windshield (dark)
    ws = ET.SubElement(link, "visual", name="windshield")
    geom = ET.SubElement(ws, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{0.05:.2f} {w * 0.85:.2f} {h * 0.35:.2f}"
    ET.SubElement(ws, "pose").text = f"{l * 0.2:.2f} 0 {h * 0.82:.2f} 0 0 0"
    _mat(ws, 0.2, 0.3, 0.4, 0.7)

    # Wheels (4 cylinders)
    wh = h * 0.3
    for wi, (wx_off, wy_off) in enumerate([
        (l * 0.3, w * 0.5), (l * 0.3, -w * 0.5),
        (-l * 0.3, w * 0.5), (-l * 0.3, -w * 0.5),
    ]):
        wheel = ET.SubElement(link, "visual", name=f"wheel_{wi}")
        geom = ET.SubElement(wheel, "geometry")
        cyl = ET.SubElement(geom, "cylinder")
        ET.SubElement(cyl, "radius").text = f"{wh / 2:.3f}"
        ET.SubElement(cyl, "length").text = f"{0.15:.3f}"
        ET.SubElement(wheel, "pose").text = f"{wx_off:.2f} {wy_off:.2f} {wh / 2:.3f} {math.pi / 2:.4f} 0 0"
        _mat(wheel, 0.12, 0.12, 0.12)


def _add_concealment_markers(
    world: ET.Element,
    clusters: list[tuple[float, float, float]],
    terrain: TerrainInfo,
) -> None:
    for i, (cx, cy, _cr) in enumerate(clusters):
        model = ET.SubElement(world, "model", name=f"concealment_zone_{i}")
        ET.SubElement(model, "static").text = "true"
        z = terrain_z(terrain, cx, cy)
        ET.SubElement(model, "pose").text = f"{cx:.2f} {cy:.2f} {z + 0.1:.2f} 0 0 0"
        link = ET.SubElement(model, "link", name="link")
        vis = ET.SubElement(link, "visual", name="marker")
        geom = ET.SubElement(vis, "geometry")
        cyl = ET.SubElement(geom, "cylinder")
        ET.SubElement(cyl, "radius").text = "0.3"
        ET.SubElement(cyl, "length").text = "0.05"
        m = ET.SubElement(vis, "material")
        ET.SubElement(m, "ambient").text = "1 0 0 0.3"
        ET.SubElement(m, "diffuse").text = "1 0 0 0.3"


def _sdf_header(world_name: str, shadows: bool) -> tuple[ET.Element, ET.Element]:
    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name=world_name)

    ET.SubElement(world, "gravity").text = "0 0 -9.8066"

    sph = ET.SubElement(world, "spherical_coordinates")
    for tag, val in [
        ("surface_model", "EARTH_WGS84"),
        ("latitude_deg", "36.38591274143771"),
        ("longitude_deg", "-6.03973388671875"),
        ("elevation", "24.80000000000109"),
    ]:
        ET.SubElement(sph, tag).text = val

    physics = ET.SubElement(world, "physics", type="ode")
    ET.SubElement(physics, "max_step_size").text = "0.004"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    ET.SubElement(physics, "real_time_update_rate").text = "250"

    light = ET.SubElement(world, "light", type="directional", name="sun")
    ET.SubElement(light, "cast_shadows").text = "true" if shadows else "false"
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "specular").text = "0.2 0.2 0.2 1"
    att = ET.SubElement(light, "attenuation")
    ET.SubElement(att, "range").text = "1000"
    ET.SubElement(att, "constant").text = "0.9"
    ET.SubElement(att, "linear").text = "0.01"
    ET.SubElement(att, "quadratic").text = "0.001"
    ET.SubElement(light, "direction").text = "-0.5 0.1 -0.9"

    return sdf, world


# ---------------------------------------------------------------------------
# Z placement audit
# ---------------------------------------------------------------------------

def audit_placements(
    terrain: TerrainInfo,
    placements: list[Placement],
    people: list[Placement],
    vehicles: list[Placement],
) -> list[str]:
    """Check for floating or sunken entities. Returns list of warnings."""
    warnings = []
    for p in placements + people + vehicles:
        expected_z = terrain_z(terrain, p.x, p.y)
        z_offset = p.model_type
        # For people actors, the actor mesh stands at Z + 1.0
        # The static inline marker should be at ground Z
        if p.model_type == "person":
            # Static marker should have Z = ground level
            if abs(p.z - expected_z) > 0.5:
                warnings.append(
                    f"WARNING: {p.semantic_name} at ({p.x:.1f}, {p.y:.1f}) "
                    f"Z={p.z:.2f} but terrain Z={expected_z:.2f} (diff={p.z - expected_z:.2f})"
                )
        elif p.model_type == "vehicle":
            # Vehicles should be at ground level
            if abs(p.z - expected_z) > 0.5:
                warnings.append(
                    f"WARNING: {p.semantic_name} at ({p.x:.1f}, {p.y:.1f}) "
                    f"Z={p.z:.2f} but terrain Z={expected_z:.2f} (diff={p.z - expected_z:.2f})"
                )
        else:
            # Vegetation: roots at ground level
            if abs(p.z - expected_z) > 1.0:
                warnings.append(
                    f"WARNING: {p.model_type} at ({p.x:.1f}, {p.y:.1f}) "
                    f"Z={p.z:.2f} but terrain Z={expected_z:.2f} (diff={p.z - expected_z:.2f})"
                )
    return warnings


# ---------------------------------------------------------------------------
# Generate world
# ---------------------------------------------------------------------------

def generate_world_sdf(
    terrain_name: str,
    density: str,
    seed: int,
    exclusion_zones: list[tuple[float, float, float]] | None = None,
) -> tuple[str, list[tuple[float, float, float]]]:
    """Generate complete SDF world. Returns (xml, clusters)."""
    terrain = load_terrain(terrain_name)
    profile = DENSITY_PROFILES[density]

    world_name = f"{terrain_name}_veg_{density}"
    sdf, world = _sdf_header(world_name, profile["shadows"])

    # Terrain + helipad
    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "uri").text = f"model://{terrain_name}"
    inc2 = ET.SubElement(world, "include")
    ET.SubElement(inc2, "uri").text = "model://helipad"

    placements, people, vehicles, clusters = generate_placements(
        terrain, density, seed, exclusion_zones,
    )

    # Audit
    warnings = audit_placements(terrain, placements, people, vehicles)
    for w in warnings:
        print(f"  {w}")

    # Add entities to world
    idx = 0
    for p in placements:
        _add_inline_vegetation(world, idx, p)
        idx += 1

    for i, p in enumerate(people):
        _add_person_actor(world, p, i, random.Random(seed + i))

    for i, p in enumerate(vehicles):
        _add_vehicle_inline(world, p, i)

    _add_concealment_markers(world, clusters, terrain)

    ET.indent(sdf, space="  ")
    xml_str = '<?xml version="1.0" ?>\n' + ET.tostring(sdf, encoding="unicode")
    return xml_str, clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Place vegetation, people, vehicles on real-terrain world")
    parser.add_argument("--world", required=True, help="World/model name (e.g. 1779343687303)")
    parser.add_argument("--density", choices=["light", "medium", "heavy", "camera_lite", "realistic_lite"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Generate all three densities")
    parser.add_argument("--audit", action="store_true", help="Audit existing world file for Z issues")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    densities = ["light", "medium", "heavy", "camera_lite", "realistic_lite"] if args.all else [args.density]
    out_dir = Path(args.output_dir) if args.output_dir else GZ_WORLDS_DIR

    for density in densities:
        print(f"\nGenerating {args.world}_veg_{density}.sdf ...")
        xml_str, clusters = generate_world_sdf(args.world, density, args.seed)

        out_path = out_dir / f"{args.world}_veg_{density}.sdf"
        out_path.write_text(xml_str)
        n_veg = xml_str.count('<model name="veg_')
        n_people = xml_str.count("<actor")
        n_person_models = xml_str.count('<model name="person_')
        n_vehicles = xml_str.count('<model name="') - n_veg - n_person_models - xml_str.count('<model name="concealment_')
        print(f"  Written: {out_path}")
        print(f"  Vegetation: {n_veg}")
        print(f"  People actors: {n_people} (distinct models: {n_person_models})")
        print(f"  Vehicles: {n_vehicles}")
        print(f"  Concealment clusters: {len(clusters)}")
        for i, (cx, cy, cr) in enumerate(clusters):
            print(f"    Cluster {i}: ({cx:.1f}, {cy:.1f}) r={cr:.1f}m")

    print("\nDone.")


if __name__ == "__main__":
    main()
