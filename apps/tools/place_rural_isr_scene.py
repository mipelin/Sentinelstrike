"""Generate rural ISR training worlds with concealment-aware target placement.

Creates realistic rural/countryside scenes on existing real-terrain Gazebo worlds
for ISR detection, tracking, ReID, occlusion, and standoff testing.

Two target categories:
  - Civilian: walking on roads, standing in open areas, parked vehicles
  - Military-like: near tree lines, concealed, stop-and-go behavior

Usage:
    python -m apps.tools.place_rural_isr_scene --world 1779343687303 --density medium
    python -m apps.tools.place_rural_isr_scene --world 1779343687303 --all
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from apps.tools.place_real_terrain_vegetation import (
    VEG_GEOMETRY,
    SCALE_RANGES,
    TerrainInfo,
    _add_concealment_markers,
    _add_inline_vegetation,
    _sdf_header,
    generate_placements,
    load_terrain,
    terrain_z,
)
from apps.tools.import_fuel_isr_assets import (
    ISR_ACTOR_SKINS,
    ISR_FUEL_VEHICLES,
    ISR_STATIC_PEOPLE,
    verify_isr_all,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PX4_DIR = PROJECT_ROOT.parent / "PX4-Autopilot"
GZ_WORLDS_DIR = PX4_DIR / "Tools" / "simulation" / "gz" / "worlds"

FUEL_BASE = "https://fuel.gazebosim.org/1.0"


# ---------------------------------------------------------------------------
# Global name uniqueness registry
# ---------------------------------------------------------------------------

_used_names: set[str] = set()
_name_counters: dict[str, int] = {}


def _reset_names() -> None:
    _used_names.clear()
    _name_counters.clear()


def _unique_name(prefix: str) -> str:
    """Generate a globally unique SDF name with the given prefix."""
    counter = _name_counters.get(prefix, 0) + 1
    _name_counters[prefix] = counter
    name = f"{prefix}_{counter:04d}"
    while name in _used_names:
        counter += 1
        _name_counters[prefix] = counter
        name = f"{prefix}_{counter:04d}"
    _used_names.add(name)
    return name


def _claim_name(name: str) -> str:
    """Register a name as used, raising if duplicate."""
    if name in _used_names:
        raise RuntimeError(f"Duplicate SDF name: {name}")
    _used_names.add(name)
    return name


def validate_unique_sdf_names(xml_str: str) -> list[str]:
    """Parse generated SDF and return list of duplicate names found.

    Checks: model/actor/light names globally unique.
    Checks: visual/collision names unique within each link.
    """
    root = ET.fromstring(xml_str)
    dups: list[str] = []

    # Global uniqueness for model, actor, light
    seen_global: dict[str, str] = {}
    global_tags = {"model", "actor", "light"}
    for elem in root.iter():
        name = elem.get("name")
        if name is None:
            continue
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag not in global_tags:
            continue
        if name in seen_global:
            dups.append(f"<{tag}> name='{name}' (duplicate of <{seen_global[name]}>)")
        else:
            seen_global[name] = tag

    # Per-link uniqueness for visual, collision
    for model in root.iter("model"):
        model_name = model.get("name", "?")
        for link in model.iter("link"):
            per_link: set[str] = set()
            for child in link:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag in ("visual", "collision"):
                    vname = child.get("name", "")
                    if vname in per_link:
                        dups.append(f"<{tag}> name='{vname}' in model '{model_name}'")
                    else:
                        per_link.add(vname)

    return dups

# ---------------------------------------------------------------------------
# Rural density profiles
# ---------------------------------------------------------------------------

RURAL_PROFILES = {
    "light": {
        "civilian_people": 10, "military_people": 4,
        "civilian_vehicles": 6, "concealed_vehicles": 2,
        "tree_line_segments": 3, "veg_clusters": 4,
        "dirt_roads": 2, "sheds": 2,
        "shadows": False,
    },
    "medium": {
        "civilian_people": 25, "military_people": 12,
        "civilian_vehicles": 12, "concealed_vehicles": 6,
        "tree_line_segments": 6, "veg_clusters": 8,
        "dirt_roads": 3, "sheds": 4,
        "shadows": True,
    },
    "heavy": {
        "civilian_people": 60, "military_people": 30,
        "civilian_vehicles": 25, "concealed_vehicles": 12,
        "tree_line_segments": 10, "veg_clusters": 12,
        "dirt_roads": 4, "sheds": 6,
        "shadows": True,
    },
    "camera_lite": {
        "civilian_people": 4, "military_people": 4,
        "civilian_vehicles": 2, "concealed_vehicles": 1,
        "tree_line_segments": 1, "veg_clusters": 2,
        "dirt_roads": 1, "sheds": 2,
        "shadows": False,
    },
    "realistic_lite": {
        "civilian_people": 8, "military_people": 4,
        "civilian_vehicles": 2, "concealed_vehicles": 1,
        "tree_line_segments": 2, "veg_clusters": 3,
        "dirt_roads": 1, "sheds": 2,
        "shadows": False,
    },
}

# ---------------------------------------------------------------------------
# Clothing palettes for re-ID variation
# ---------------------------------------------------------------------------

CIVILIAN_UPPER_COLORS = [
    "red", "blue", "white", "yellow", "orange", "teal",
    "pink", "lime", "purple", "cream", "coral", "sky_blue",
]
CIVILIAN_LOWER_COLORS = [
    "blue", "khaki", "gray", "brown", "black", "navy",
    "dark_green", "beige",
]
MILITARY_UPPER_COLORS = [
    "dark_green", "olive", "brown", "khaki", "gray", "dark_gray",
    "tan", "forest_green",
]
MILITARY_LOWER_COLORS = [
    "dark_green", "olive", "brown", "khaki", "dark_gray", "tan",
]

CLOTHING_RGB: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.15, 0.15), "blue": (0.15, 0.25, 0.85),
    "white": (0.92, 0.92, 0.92), "yellow": (0.90, 0.85, 0.15),
    "orange": (0.90, 0.55, 0.10), "teal": (0.10, 0.70, 0.65),
    "pink": (0.90, 0.55, 0.65), "lime": (0.55, 0.90, 0.15),
    "purple": (0.55, 0.20, 0.70), "cream": (0.92, 0.88, 0.75),
    "coral": (0.90, 0.45, 0.35), "sky_blue": (0.40, 0.65, 0.90),
    "khaki": (0.75, 0.70, 0.50), "gray": (0.55, 0.55, 0.55),
    "brown": (0.55, 0.38, 0.22), "black": (0.15, 0.15, 0.15),
    "navy": (0.12, 0.15, 0.40), "dark_green": (0.20, 0.40, 0.15),
    "beige": (0.82, 0.78, 0.65), "olive": (0.45, 0.50, 0.25),
    "dark_gray": (0.30, 0.30, 0.30), "tan": (0.72, 0.62, 0.42),
    "forest_green": (0.15, 0.45, 0.15),
}

SPEED_RANGES = {
    "slow": (5.0, 8.0), "normal": (3.0, 6.0), "fast": (2.0, 4.0),
    "crawl": (8.0, 14.0), "dash": (1.5, 3.0),
}

# ---------------------------------------------------------------------------
# Placement dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RuralPerson:
    x: float
    y: float
    z: float
    yaw: float
    semantic_name: str
    target_type: str  # "civilian" | "military_like"
    placement_type: str  # "walking" | "standing"
    behavior: str  # "road_walk" | "open_loiter" | "tree_line_crawl" | "stop_and_go" | "static"
    skin_url: str = ""
    static_uri: str = ""
    z_offset: float = 0.0
    marker_color: tuple[float, float, float] = (1.0, 1.0, 0.0)
    speed_class: str = "normal"
    height_scale: float = 1.0
    concealment_level: float = 0.0
    upper_color: str = ""
    lower_color: str = ""
    target_id: str = ""
    difficulty: str = "easy"


@dataclass
class RuralVehicle:
    x: float
    y: float
    z: float
    yaw: float
    semantic_name: str
    target_type: str  # "civilian_vehicle" | "concealed_vehicle"
    fuel_uri: str
    z_offset: float = 0.0
    marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    concealment_level: float = 0.0
    target_id: str = ""
    difficulty: str = "easy"
    moving: bool = False


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------


def terrain_slope(terrain: TerrainInfo, x: float, y: float, radius: float = 2.0) -> float:
    z_c = terrain_z(terrain, x, y)
    z_xp = terrain_z(terrain, x + radius, y)
    z_xn = terrain_z(terrain, x - radius, y)
    z_yp = terrain_z(terrain, x, y + radius)
    z_yn = terrain_z(terrain, x, y - radius)
    dx = (z_xp - z_xn) / (2 * radius)
    dy = (z_yp - z_yn) / (2 * radius)
    return math.atan(math.sqrt(dx * dx + dy * dy))


def _terrain_bounds(terrain: TerrainInfo) -> tuple[float, float, float, float]:
    return (
        terrain.offset_x - terrain.width_m / 2,
        terrain.offset_y - terrain.height_m / 2,
        terrain.offset_x + terrain.width_m / 2,
        terrain.offset_y + terrain.height_m / 2,
    )


_BOUNDS_MARGIN = 20.0


def _has_real_bounds(terrain: TerrainInfo) -> bool:
    return terrain.width_m > 0 and terrain.height_m > 0


def _clamp_xy(x: float, y: float, terrain: TerrainInfo,
              margin: float = _BOUNDS_MARGIN) -> tuple[float, float]:
    if not _has_real_bounds(terrain):
        return x, y
    x_min, y_min, x_max, y_max = _terrain_bounds(terrain)
    x = max(x_min + margin, min(x, x_max - margin))
    y = max(y_min + margin, min(y, y_max - margin))
    return x, y


def _in_bounds(x: float, y: float, terrain: TerrainInfo,
               margin: float = _BOUNDS_MARGIN) -> bool:
    if not _has_real_bounds(terrain):
        return True
    x_min, y_min, x_max, y_max = _terrain_bounds(terrain)
    return (x_min + margin <= x <= x_max - margin
            and y_min + margin <= y <= y_max - margin)


# ---------------------------------------------------------------------------
# Dirt road generation
# ---------------------------------------------------------------------------


def _generate_dirt_roads(
    terrain: TerrainInfo, n_roads: int, rng: random.Random,
    exclusion_zones: list[tuple[float, float, float]],
    density: str = "",
) -> list[list[tuple[float, float]]]:
    """Generate dirt road paths as sequences of (x, y) waypoints."""
    margin = 30.0

    roads: list[list[tuple[float, float]]] = []
    for _ in range(n_roads):
        # Random start within bounds
        if density == "realistic_lite":
            x0 = rng.uniform(170.0, 270.0)
            y0 = rng.uniform(-400.0, -300.0)
            x1 = rng.uniform(170.0, 270.0)
            y1 = rng.uniform(-400.0, -300.0)
        else:
            x0, y0 = _clamp_xy(
                rng.gauss(terrain.offset_x, terrain.width_m * 0.2),
                rng.gauss(terrain.offset_y, terrain.height_m * 0.2),
                terrain, margin,
            )
            # Random end within bounds
            x1, y1 = _clamp_xy(
                rng.gauss(terrain.offset_x, terrain.width_m * 0.2),
                rng.gauss(terrain.offset_y, terrain.height_m * 0.2),
                terrain, margin,
            )

        # Subdivide with slight jitter for natural curves
        n_segments = rng.randint(4, 8)
        points = []
        for j in range(n_segments + 1):
            t = j / n_segments
            px = x0 + (x1 - x0) * t + rng.uniform(-5, 5)
            py = y0 + (y1 - y0) * t + rng.uniform(-5, 5)
            px, py = _clamp_xy(px, py, terrain, margin)
            if density == "realistic_lite":
                px = max(170.0, min(px, 270.0))
                py = max(-400.0, min(py, -300.0))
            points.append((px, py))
        roads.append(points)

    return roads


def _add_dirt_road(
    world: ET.Element, terrain: TerrainInfo,
    points: list[tuple[float, float]], road_idx: int, rng: random.Random,
) -> None:
    """Add a dirt road as flat brown strips between waypoints."""
    road_width = 3.5
    brown = (0.55, 0.42, 0.30)

    for j in range(len(points) - 1):
        x0, y0 = points[j]
        x1, y1 = points[j + 1]
        if not _in_bounds(x0, y0, terrain) or not _in_bounds(x1, y1, terrain):
            PLACEMENT_DIAGNOSTICS["clipped_road_segments"] += 1
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        length = math.hypot(x1 - x0, y1 - y0)
        angle = math.atan2(y1 - y0, x1 - x0)
        z = terrain_z(terrain, cx, cy) + 0.02

        model_name = _unique_name("road")
        model = ET.SubElement(world, "model", name=model_name)
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = f"{cx:.2f} {cy:.2f} {z:.2f} 0 0 {angle:.2f}"
        link = ET.SubElement(model, "link", name="link")
        vis = ET.SubElement(link, "visual", name="road")
        geom = ET.SubElement(vis, "geometry")
        bx = ET.SubElement(geom, "box")
        ET.SubElement(bx, "size").text = f"{length:.2f} {road_width:.2f} 0.02"
        mat = ET.SubElement(vis, "material")
        r, g, b = brown
        noise = rng.uniform(-0.03, 0.03)
        ET.SubElement(mat, "ambient").text = f"{r + noise:.2f} {g + noise:.2f} {b + noise:.2f} 1"
        ET.SubElement(mat, "diffuse").text = f"{r + noise:.2f} {g + noise:.2f} {b + noise:.2f} 1"


# ---------------------------------------------------------------------------
# Shed generation
# ---------------------------------------------------------------------------


def _add_shed(
    world: ET.Element, terrain: TerrainInfo,
    x: float, y: float, yaw: float, shed_idx: int, rng: random.Random,
) -> None:
    """Add a small rural shed/structure."""
    z = terrain_z(terrain, x, y)
    w = rng.uniform(3.0, 5.0)
    h = rng.uniform(2.0, 3.0)
    d = rng.uniform(3.0, 5.0)
    wall_rgb = (0.65, 0.58, 0.48)
    roof_rgb = (0.45, 0.30, 0.20)

    model_name = _unique_name("shed")
    model = ET.SubElement(world, "model", name=model_name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.2f}"
    link = ET.SubElement(model, "link", name="link")

    # Walls
    vis = ET.SubElement(link, "visual", name="walls")
    geom = ET.SubElement(vis, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{w:.2f} {d:.2f} {h:.2f}"
    ET.SubElement(vis, "pose").text = f"0 0 {h / 2:.2f} 0 0 0"
    mat = ET.SubElement(vis, "material")
    ET.SubElement(mat, "ambient").text = f"{wall_rgb[0]:.2f} {wall_rgb[1]:.2f} {wall_rgb[2]:.2f} 1"
    ET.SubElement(mat, "diffuse").text = f"{wall_rgb[0]:.2f} {wall_rgb[1]:.2f} {wall_rgb[2]:.2f} 1"

    # Roof (slightly larger, flat)
    vis2 = ET.SubElement(link, "visual", name="roof")
    geom2 = ET.SubElement(vis2, "geometry")
    bx2 = ET.SubElement(geom2, "box")
    ET.SubElement(bx2, "size").text = f"{w + 0.4:.2f} {d + 0.4:.2f} 0.15"
    ET.SubElement(vis2, "pose").text = f"0 0 {h + 0.08:.2f} 0 0 0"
    mat2 = ET.SubElement(vis2, "material")
    ET.SubElement(mat2, "ambient").text = f"{roof_rgb[0]:.2f} {roof_rgb[1]:.2f} {roof_rgb[2]:.2f} 1"
    ET.SubElement(mat2, "diffuse").text = f"{roof_rgb[0]:.2f} {roof_rgb[1]:.2f} {roof_rgb[2]:.2f} 1"


# ---------------------------------------------------------------------------
# Concealment zone generation (tree lines)
# ---------------------------------------------------------------------------


def _generate_tree_lines(
    terrain: TerrainInfo, n_segments: int, rng: random.Random,
    exclusion_zones: list[tuple[float, float, float]],
    density: str = "",
) -> list[list[tuple[float, float, float]]]:
    """Generate tree lines as lists of (x, y, radius) concealment zones."""
    margin = 25.0

    zones: list[list[tuple[float, float, float]]] = []
    for _ in range(n_segments):
        # Random start clamped to terrain
        x0, y0 = _random_pos(terrain, rng, exclusion_zones, margin, density)
        length = rng.uniform(20, 60) if density != "realistic_lite" else rng.uniform(10, 30)
        angle = rng.uniform(0, 2 * math.pi)
        x1, y1 = _clamp_xy(x0 + length * math.cos(angle),
                            y0 + length * math.sin(angle),
                            terrain, margin)
        if density == "realistic_lite":
            x1 = max(170.0, min(x1, 270.0))
            y1 = max(-400.0, min(y1, -300.0))

        # Subdivide into concealment points
        n_pts = max(3, int(length / 5))
        line_zones: list[tuple[float, float, float]] = []
        for j in range(n_pts + 1):
            t = j / n_pts
            px = x0 + (x1 - x0) * t + rng.uniform(-1.5, 1.5)
            py = y0 + (y1 - y0) * t + rng.uniform(-1.5, 1.5)
            px, py = _clamp_xy(px, py, terrain, margin)
            if density == "realistic_lite":
                px = max(170.0, min(px, 270.0))
                py = max(-400.0, min(py, -300.0))
            skip = any(math.hypot(px - ex, py - ey) < er for ex, ey, er in exclusion_zones)
            if not skip:
                line_zones.append((px, py, rng.uniform(3, 6)))
        if line_zones:
            zones.append(line_zones)

    return zones


# ---------------------------------------------------------------------------
# Placement generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Placement helpers (module-level so generate_rural_world_sdf can use them)
# ---------------------------------------------------------------------------

PLACEMENT_DIAGNOSTICS: dict[str, int] = {
    "attempts": 0, "rejections_exclusion": 0, "rejections_slope": 0,
    "accepted_flat": 0, "accepted_random": 0, "fallback_used": 0,
    "outside_bounds_rejections": 0, "clipped_road_segments": 0,
    "failed_shed_placements": 0,
}


def _reset_diagnostics() -> None:
    for key in PLACEMENT_DIAGNOSTICS:
        PLACEMENT_DIAGNOSTICS[key] = 0


def _in_exclusion(x: float, y: float,
                  exclusion_zones: list[tuple[float, float, float]]) -> bool:
    return any(math.hypot(x - ex, y - ey) < er for ex, ey, er in exclusion_zones)


def _random_pos(
    terrain: TerrainInfo,
    rng: random.Random,
    exclusion_zones: list[tuple[float, float, float]],
    margin: float = 20.0,
    density: str = "",
) -> tuple[float, float]:
    """Random position within terrain bounds, avoiding exclusion zones."""
    if density == "realistic_lite":
        x_min, x_max = 170.0, 270.0
        y_min, y_max = -400.0, -300.0
    else:
        x_min, y_min, x_max, y_max = _terrain_bounds(terrain)
        x_min += margin
        x_max -= margin
        y_min += margin
        y_max -= margin

    for _ in range(100):
        PLACEMENT_DIAGNOSTICS["attempts"] += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if not _in_exclusion(x, y, exclusion_zones):
            PLACEMENT_DIAGNOSTICS["accepted_random"] += 1
            return x, y
        PLACEMENT_DIAGNOSTICS["rejections_exclusion"] += 1

    PLACEMENT_DIAGNOSTICS["fallback_used"] += 1
    return rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)


def _flat_pos(
    terrain: TerrainInfo,
    rng: random.Random,
    exclusion_zones: list[tuple[float, float, float]],
    max_slope_deg: float = 15.0,
    margin: float = 20.0,
    density: str = "",
) -> tuple[float, float]:
    """Random position on flat ground (< max_slope_deg), avoiding exclusion zones."""
    for _ in range(200):
        PLACEMENT_DIAGNOSTICS["attempts"] += 1
        x, y = _random_pos(terrain, rng, exclusion_zones, margin, density)
        if terrain_slope(terrain, x, y) < math.radians(max_slope_deg):
            PLACEMENT_DIAGNOSTICS["accepted_flat"] += 1
            return x, y
        PLACEMENT_DIAGNOSTICS["rejections_slope"] += 1

    PLACEMENT_DIAGNOSTICS["fallback_used"] += 1
    return _random_pos(terrain, rng, exclusion_zones, margin, density)


def _pos_on_road(
    roads: list[list[tuple[float, float]]], rng: random.Random,
) -> tuple[float, float] | None:
    """Pick a random position on a dirt road."""
    if not roads:
        return None
    road = rng.choice(roads)
    if len(road) < 2:
        return road[0]
    j = rng.randint(0, len(road) - 2)
    t = rng.random()
    x = road[j][0] + (road[j + 1][0] - road[j][0]) * t
    y = road[j][1] + (road[j + 1][1] - road[j][1]) * t
    return x, y


def _pos_near_concealment(
    tree_lines: list[list[tuple[float, float, float]]],
    clusters: list[tuple[float, float, float]],
    rng: random.Random,
    offset_range: tuple[float, float] = (1.0, 4.0),
) -> tuple[float, float, float] | None:
    """Pick a position near a concealment zone. Returns (x, y, concealment_level)."""
    all_zones: list[tuple[float, float, float]] = []
    for line in tree_lines:
        all_zones.extend(line)
    for cx, cy, cr in clusters:
        all_zones.append((cx, cy, cr))

    if not all_zones:
        return None

    zx, zy, zr = rng.choice(all_zones)
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(offset_range[0], offset_range[1])
    x = zx + dist * math.cos(angle)
    y = zy + dist * math.sin(angle)
    concealment = max(0.3, 1.0 - dist / zr)
    return x, y, concealment


# Registry for startup validation
_PLACEMENT_HELPERS = {
    "random_pos": _random_pos,
    "flat_pos": _flat_pos,
    "pos_on_road": _pos_on_road,
    "pos_near_concealment": _pos_near_concealment,
}


def validate_placement_helpers() -> None:
    """Ensure all placement helpers are callable. Raises RuntimeError on mismatch."""
    missing = [name for name, fn in _PLACEMENT_HELPERS.items() if not callable(fn)]
    if missing:
        raise RuntimeError(f"Placement helpers missing or not callable: {missing}")


# ---------------------------------------------------------------------------
# Placement generation
# ---------------------------------------------------------------------------


def generate_rural_placements(
    terrain: TerrainInfo,
    density: str,
    seed: int,
    exclusion_zones: list[tuple[float, float, float]] | None = None,
) -> tuple[
    list[RuralPerson],
    list[RuralVehicle],
    list[list[tuple[float, float]]],
    list[list[tuple[float, float, float]]],
]:
    """Generate all placements for a rural ISR scene."""
    validate_placement_helpers()
    _reset_diagnostics()

    rng = random.Random(seed)
    profile = RURAL_PROFILES[density]

    if exclusion_zones is None:
        exclusion_zones = [(0.0, 0.0, 15.0)]

    # Generate dirt roads
    roads = _generate_dirt_roads(terrain, profile["dirt_roads"], rng, exclusion_zones, density)

    # Generate vegetation clusters via existing system
    _, _, _, clusters = generate_placements(terrain, density, seed, exclusion_zones)

    # Generate tree line concealment zones
    tree_lines = _generate_tree_lines(terrain, profile["tree_line_segments"], rng, exclusion_zones, density)

    people: list[RuralPerson] = []
    vehicles: list[RuralVehicle] = []

    # --- Civilian people ---
    for i in range(profile["civilian_people"]):
        upper = rng.choice(CIVILIAN_UPPER_COLORS)
        lower = rng.choice(CIVILIAN_LOWER_COLORS)
        marker_rgb = CLOTHING_RGB[upper]

        # Place on road or open area
        if density == "realistic_lite" and i == 0:
            x, y = 220.0, -350.0
            road_pos = None
        else:
            road_pos = _pos_on_road(roads, rng) if rng.random() < 0.7 else None
            if road_pos:
                x, y = road_pos
            else:
                x, y = _random_pos(terrain, rng, exclusion_zones, density=density)

        z = terrain_z(terrain, x, y)
        yaw = rng.uniform(0, 2 * math.pi)
        if density in ("camera_lite", "realistic_lite"):
            is_walking = False
        else:
            is_walking = rng.random() < 0.6
        tid = f"CIV_{i + 1:03d}"

        if is_walking:
            skin = ISR_ACTOR_SKINS[i % len(ISR_ACTOR_SKINS)]
            people.append(RuralPerson(
                x, y, z, yaw, f"civilian_walking_{i:03d}", "civilian", "walking",
                behavior="road_walk" if road_pos else "open_loiter",
                skin_url=skin, z_offset=0.0,
                marker_color=marker_rgb,
                speed_class=rng.choice(["normal", "slow", "fast"]),
                height_scale=rng.uniform(0.9, 1.1),
                concealment_level=0.0,
                upper_color=upper, lower_color=lower,
                target_id=tid, difficulty="easy",
            ))
        else:
            static = ISR_STATIC_PEOPLE[i % len(ISR_STATIC_PEOPLE)]
            people.append(RuralPerson(
                x, y, z, yaw, f"civilian_standing_{i:03d}", "civilian", "standing",
                behavior="static",
                static_uri=static["uri"], z_offset=static["z_off"],
                marker_color=marker_rgb,
                concealment_level=0.0,
                upper_color=upper, lower_color=lower,
                target_id=tid, difficulty="easy",
            ))

    # --- Military-like people ---
    for i in range(profile["military_people"]):
        upper = rng.choice(MILITARY_UPPER_COLORS)
        lower = rng.choice(MILITARY_LOWER_COLORS)
        marker_rgb = CLOTHING_RGB[upper]

        # Place near concealment
        if density == "realistic_lite" and i == 0:
            x, y = 218.8, -351.2
            concealment = 0.4
        else:
            conceal_pos = _pos_near_concealment(tree_lines, clusters, rng)
            if conceal_pos:
                x, y, concealment = conceal_pos
                x, y = _clamp_xy(x, y, terrain)
                if density == "realistic_lite":
                    x = max(170.0, min(x, 270.0))
                    y = max(-400.0, min(y, -300.0))
            else:
                x, y = _random_pos(terrain, rng, exclusion_zones, density=density)
                concealment = 0.0

        z = terrain_z(terrain, x, y)
        yaw = rng.uniform(0, 2 * math.pi)
        behavior = rng.choice(["tree_line_crawl", "stop_and_go", "static"])
        if density in ("camera_lite", "realistic_lite"):
            is_walking = False
        else:
            is_walking = behavior != "static" if density != "camera_lite" else False
        tid = f"MIL_{i + 1:03d}"

        difficulty = "medium" if concealment < 0.5 else "hard"

        if is_walking:
            skin = ISR_ACTOR_SKINS[i % len(ISR_ACTOR_SKINS)]
            people.append(RuralPerson(
                x, y, z, yaw, f"military_{i:03d}", "military_like", "walking",
                behavior=behavior,
                skin_url=skin, z_offset=0.0,
                marker_color=marker_rgb,
                speed_class="crawl" if behavior == "tree_line_crawl" else "slow",
                height_scale=rng.uniform(0.9, 1.1),
                concealment_level=concealment,
                upper_color=upper, lower_color=lower,
                target_id=tid, difficulty=difficulty,
            ))
        else:
            static = ISR_STATIC_PEOPLE[i % len(ISR_STATIC_PEOPLE)]
            people.append(RuralPerson(
                x, y, z, yaw, f"military_static_{i:03d}", "military_like", "standing",
                behavior="static",
                static_uri=static["uri"], z_offset=static["z_off"],
                marker_color=marker_rgb,
                concealment_level=concealment,
                upper_color=upper, lower_color=lower,
                target_id=tid, difficulty=difficulty,
            ))

    # --- Civilian vehicles ---
    for i in range(profile["civilian_vehicles"]):
        if density in ("camera_lite", "realistic_lite"):
            # Avoid hatchback, bus, and prius with broken textures or sensors
            safe_vehicles = [v for v in ISR_FUEL_VEHICLES if v["key"] in ("pickup", "truckbox")]
            fuel_v = safe_vehicles[i % len(safe_vehicles)]
        else:
            fuel_v = ISR_FUEL_VEHICLES[i % len(ISR_FUEL_VEHICLES)]
        if density == "realistic_lite" and i == 0:
            x, y = 221.2, -348.8
        else:
            x, y = _flat_pos(terrain, rng, exclusion_zones, density=density)
            # Prefer near roads
            if rng.random() < 0.6:
                road_pos = _pos_on_road(roads, rng)
                if road_pos:
                    rx, ry = road_pos
                    angle = rng.uniform(0, 2 * math.pi)
                    x, y = _clamp_xy(rx + 4.0 * math.cos(angle),
                                      ry + 4.0 * math.sin(angle), terrain)
                    if density == "realistic_lite":
                        x = max(170.0, min(x, 270.0))
                        y = max(-400.0, min(y, -300.0))

        z = terrain_z(terrain, x, y)
        yaw = rng.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2]) + rng.uniform(-0.15, 0.15)
        tid = f"VEH_CIV_{i + 1:03d}"

        vehicles.append(RuralVehicle(
            x, y, z, yaw, f"vehicle_civ_{fuel_v['key']}_{i:03d}",
            "civilian_vehicle",
            fuel_uri=fuel_v["uri"], z_offset=fuel_v["z_off"],
            marker_color=CLOTHING_RGB["white"],
            concealment_level=0.0,
            target_id=tid, difficulty="easy",
            moving=rng.random() < 0.2 if density not in ("camera_lite", "realistic_lite") else False,
        ))

    # --- Concealed vehicles ---
    for i in range(profile["concealed_vehicles"]):
        if density in ("camera_lite", "realistic_lite"):
            fuel_v = ISR_FUEL_VEHICLES[1] # pickup (known good)
        else:
            fuel_v = ISR_FUEL_VEHICLES[rng.choice([1, 2])]
        conceal_pos = _pos_near_concealment(tree_lines, clusters, rng, offset_range=(0.5, 3.0))
        if conceal_pos:
            x, y, concealment = conceal_pos
            x, y = _clamp_xy(x, y, terrain)
            if density == "realistic_lite":
                x = max(170.0, min(x, 270.0))
                y = max(-400.0, min(y, -300.0))
        else:
            x, y = _flat_pos(terrain, rng, exclusion_zones, density=density)
            concealment = 0.0

        z = terrain_z(terrain, x, y)
        yaw = rng.uniform(0, 2 * math.pi)
        tid = f"VEH_MIL_{i + 1:03d}"

        vehicles.append(RuralVehicle(
            x, y, z, yaw, f"vehicle_concealed_{i:03d}",
            "concealed_vehicle",
            fuel_uri=fuel_v["uri"], z_offset=fuel_v["z_off"],
            marker_color=CLOTHING_RGB["dark_green"],
            concealment_level=concealment,
            target_id=tid, difficulty="hard",
        ))

    return people, vehicles, roads, tree_lines


# ---------------------------------------------------------------------------
# SDF generation helpers
# ---------------------------------------------------------------------------


def _add_reid_marker(world: ET.Element, x: float, y: float, z: float,
                     color: tuple[float, float, float], name: str,
                     marker_z_off: float = 2.0, size: float = 0.2) -> None:
    model_name = _claim_name(f"marker_{name}")
    model = ET.SubElement(world, "model", name=model_name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x:.2f} {y:.2f} {z + marker_z_off:.2f} 0 0 0"
    link = ET.SubElement(model, "link", name="link")
    vis = ET.SubElement(link, "visual", name="vis")
    geom = ET.SubElement(vis, "geometry")
    bx = ET.SubElement(geom, "box")
    ET.SubElement(bx, "size").text = f"{size:.2f} {size:.2f} 0.05"
    m = ET.SubElement(vis, "material")
    r, g, b = color
    ET.SubElement(m, "ambient").text = f"{r:.2f} {g:.2f} {b:.2f} 1"
    ET.SubElement(m, "diffuse").text = f"{r:.2f} {g:.2f} {b:.2f} 1"


def _target_metadata_comment(rp: RuralPerson | RuralVehicle) -> str:
    if isinstance(rp, RuralPerson):
        sig = f"{rp.upper_color}/{rp.lower_color}"
    else:
        sig = rp.fuel_uri.split("/")[-1]
    meta = {
        "target_id": rp.target_id,
        "target_type": rp.target_type,
        "concealment_level": round(rp.concealment_level, 2),
        "behavior_type": rp.behavior if isinstance(rp, RuralPerson) else ("moving" if rp.moving else "parked"),
        "expected_difficulty": rp.difficulty,
        "visual_signature": sig,
    }
    return f'TARGET_METADATA: {json.dumps(meta)}'


def _add_fuel_vehicle_include(world: ET.Element, terrain: TerrainInfo,
                              vp: RuralVehicle) -> None:
    z = terrain_z(terrain, vp.x, vp.y) + vp.z_offset
    comment = ET.Comment(f" {_target_metadata_comment(vp)} ")
    world.append(comment)

    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "name").text = vp.semantic_name
    ET.SubElement(inc, "pose").text = f"{vp.x:.2f} {vp.y:.2f} {z:.2f} 0 0 {vp.yaw:.2f}"
    ET.SubElement(inc, "uri").text = vp.fuel_uri
    _add_reid_marker(world, vp.x, vp.y, z, vp.marker_color,
                     vp.semantic_name, marker_z_off=2.5, size=0.35)


def _add_standing_person(world: ET.Element, terrain: TerrainInfo,
                         pp: RuralPerson) -> None:
    z = terrain_z(terrain, pp.x, pp.y) + pp.z_offset
    comment = ET.Comment(f" {_target_metadata_comment(pp)} ")
    world.append(comment)

    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "name").text = pp.semantic_name
    ET.SubElement(inc, "pose").text = f"{pp.x:.2f} {pp.y:.2f} {z:.2f} 0 0 {pp.yaw:.2f}"
    ET.SubElement(inc, "uri").text = pp.static_uri
    _add_reid_marker(world, pp.x, pp.y, z, pp.marker_color,
                     pp.semantic_name, marker_z_off=1.8, size=0.15)


def _add_walking_actor(
    world: ET.Element, terrain: TerrainInfo,
    pp: RuralPerson, rng: random.Random,
) -> None:
    comment = ET.Comment(f" {_target_metadata_comment(pp)} ")
    world.append(comment)

    actor = ET.SubElement(world, "actor", name=pp.semantic_name)

    skin = ET.SubElement(actor, "skin")
    ET.SubElement(skin, "filename").text = pp.skin_url
    ET.SubElement(skin, "scale").text = f"{pp.height_scale:.2f}"

    anim = ET.SubElement(actor, "animation", name="walk")
    ET.SubElement(anim, "filename").text = pp.skin_url
    ET.SubElement(anim, "interpolate_x").text = "true"

    script = ET.SubElement(actor, "script")
    ET.SubElement(script, "loop").text = "true"
    ET.SubElement(script, "delay_start").text = f"{rng.uniform(0, 5):.1f}"
    ET.SubElement(script, "auto_start").text = "true"

    traj = ET.SubElement(script, "trajectory", id="0", type="walk", tension="0.5")

    speed_lo, speed_hi = SPEED_RANGES.get(pp.speed_class, (3.0, 6.0))

    if pp.behavior == "tree_line_crawl":
        # Tight patrol near concealment
        radius = rng.uniform(1.5, 4.0)
        points = rng.randint(6, 10)
        for i in range(points + 1):
            t = i / points * 2 * math.pi
            wx, wy = _clamp_xy(pp.x + radius * math.cos(t),
                                pp.y + radius * math.sin(t), terrain)
            wz = terrain_z(terrain, wx, wy) + 1.0
            time_s = i * rng.uniform(speed_lo, speed_hi)
            yaw = math.atan2(wy - pp.y, wx - pp.x)
            wp = ET.SubElement(traj, "waypoint")
            ET.SubElement(wp, "time").text = f"{time_s:.1f}"
            ET.SubElement(wp, "pose").text = f"{wx:.2f} {wy:.2f} {wz:.2f} 0 0 {yaw:.2f}"
    elif pp.behavior == "stop_and_go":
        # Short dashes between cover points
        n_dashes = rng.randint(3, 5)
        dash_dist = rng.uniform(3, 8)
        time_s = 0.0
        for i in range(n_dashes):
            angle = rng.uniform(0, 2 * math.pi)
            dx, dy = dash_dist * math.cos(angle), dash_dist * math.sin(angle)
            wx, wy = _clamp_xy(pp.x + dx * (i + 1) / n_dashes,
                                pp.y + dy * (i + 1) / n_dashes, terrain)
            wz = terrain_z(terrain, wx, wy) + 1.0
            time_s += rng.uniform(speed_lo, speed_hi)
            yaw = math.atan2(dy, dx)
            wp = ET.SubElement(traj, "waypoint")
            ET.SubElement(wp, "time").text = f"{time_s:.1f}"
            ET.SubElement(wp, "pose").text = f"{wx:.2f} {wy:.2f} {wz:.2f} 0 0 {yaw:.2f}"
            time_s += rng.uniform(3.0, 8.0)
            wp2 = ET.SubElement(traj, "waypoint")
            ET.SubElement(wp2, "time").text = f"{time_s:.1f}"
            ET.SubElement(wp2, "pose").text = f"{wx:.2f} {wy:.2f} {wz:.2f} 0 0 {yaw:.2f}"
        # Return to start
        wz0 = terrain_z(terrain, pp.x, pp.y) + 1.0
        time_s += rng.uniform(speed_lo, speed_hi)
        wp3 = ET.SubElement(traj, "waypoint")
        ET.SubElement(wp3, "time").text = f"{time_s:.1f}"
        ET.SubElement(wp3, "pose").text = f"{pp.x:.2f} {pp.y:.2f} {wz0:.2f} 0 0 0"
    else:
        # Default circular patrol
        radius = rng.uniform(2.0, 6.0)
        points = rng.randint(4, 6)
        for i in range(points + 1):
            t = i / points * 2 * math.pi
            wx, wy = _clamp_xy(pp.x + radius * math.cos(t),
                                pp.y + radius * math.sin(t), terrain)
            wz = terrain_z(terrain, wx, wy) + 1.0
            time_s = i * rng.uniform(speed_lo, speed_hi)
            yaw = math.atan2(wy - pp.y, wx - pp.x)
            wp = ET.SubElement(traj, "waypoint")
            ET.SubElement(wp, "time").text = f"{time_s:.1f}"
            ET.SubElement(wp, "pose").text = f"{wx:.2f} {wy:.2f} {wz:.2f} 0 0 {yaw:.2f}"

    _add_reid_marker(world, pp.x, pp.y, terrain_z(terrain, pp.x, pp.y),
                     pp.marker_color, pp.semantic_name, marker_z_off=1.8, size=0.15)


# ---------------------------------------------------------------------------
# Tree line vegetation
# ---------------------------------------------------------------------------


def _add_tree_line_vegetation(
    world: ET.Element, terrain: TerrainInfo,
    tree_lines: list[list[tuple[float, float, float]]], rng: random.Random,
) -> None:
    """Add dense vegetation along tree lines for concealment."""
    for line in tree_lines:
        for zx, zy, zr in line:
            n_trees = max(2, int(zr))
            for _ in range(n_trees):
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(0, zr * 0.5)
                px, py = _clamp_xy(zx + dist * math.cos(angle),
                                    zy + dist * math.sin(angle), terrain)
                z = terrain_z(terrain, px, py)

                veg_type = rng.choice(["tree", "tree", "bush", "bush", "shrub"])
                geo = VEG_GEOMETRY[veg_type]
                scale_lo, scale_hi = SCALE_RANGES[veg_type]
                scale = rng.uniform(scale_lo, scale_hi)

                model_name = _unique_name(f"tl_{veg_type}")
                model = ET.SubElement(world, "model", name=model_name)
                ET.SubElement(model, "static").text = "true"
                ET.SubElement(model, "pose").text = f"{px:.2f} {py:.2f} {z:.2f} 0 0 {rng.uniform(0, 6.28):.2f}"
                link = ET.SubElement(model, "link", name="link")

                vis_count = 0

                # Trunk
                if geo["trunk_r"] > 0:
                    vis_count += 1
                    vis = ET.SubElement(link, "visual", name=f"trunk_{vis_count}")
                    geom_el = ET.SubElement(vis, "geometry")
                    cyl = ET.SubElement(geom_el, "cylinder")
                    ET.SubElement(cyl, "radius").text = f"{geo['trunk_r'] * scale:.3f}"
                    ET.SubElement(cyl, "length").text = f"{geo['trunk_h'] * scale:.2f}"
                    ET.SubElement(vis, "pose").text = f"0 0 {geo['trunk_h'] * scale / 2:.2f} 0 0 0"
                    m = ET.SubElement(vis, "material")
                    tr, tg, tb = geo["trunk_rgb"]
                    ET.SubElement(m, "ambient").text = f"{tr:.2f} {tg:.2f} {tb:.2f} 1"
                    ET.SubElement(m, "diffuse").text = f"{tr:.2f} {tg:.2f} {tb:.2f} 1"

                # Canopy
                vis_count += 1
                canopy_z = geo["canopy_z"] * scale
                vis2 = ET.SubElement(link, "visual", name=f"canopy_{vis_count}")
                geom_el2 = ET.SubElement(vis2, "geometry")
                cr = geo["canopy_r"] * scale
                if geo["canopy_type"] == "sphere":
                    sp = ET.SubElement(geom_el2, "sphere")
                    ET.SubElement(sp, "radius").text = f"{cr:.2f}"
                else:
                    cyl2 = ET.SubElement(geom_el2, "cylinder")
                    ET.SubElement(cyl2, "radius").text = f"{cr:.2f}"
                    ET.SubElement(cyl2, "length").text = f"{geo.get('canopy_h', 0.5) * scale:.2f}"
                ET.SubElement(vis2, "pose").text = f"0 0 {canopy_z:.2f} 0 0 0"
                m2 = ET.SubElement(vis2, "material")
                cr_rgb, cg_rgb, cb_rgb = geo["canopy_rgb"]
                noise = rng.uniform(-0.04, 0.04)
                ET.SubElement(m2, "ambient").text = f"{cr_rgb + noise:.2f} {cg_rgb + noise:.2f} {cb_rgb + noise:.2f} 1"
                ET.SubElement(m2, "diffuse").text = f"{cr_rgb + noise:.2f} {cg_rgb + noise:.2f} {cb_rgb + noise:.2f} 1"

                # Extra canopies
                for ec in geo.get("extra_canopies", []):
                    vis_count += 1
                    vis3 = ET.SubElement(link, "visual", name=f"extra_canopy_{vis_count}")
                    geom_el3 = ET.SubElement(vis3, "geometry")
                    sp3 = ET.SubElement(geom_el3, "sphere")
                    ET.SubElement(sp3, "radius").text = f"{ec['r'] * scale:.2f}"
                    ET.SubElement(vis3, "pose").text = (
                        f"{ec['x_off'] * scale:.2f} {ec['y_off'] * scale:.2f} "
                        f"{canopy_z + ec['z_off'] * scale:.2f} 0 0 0"
                    )
                    m3 = ET.SubElement(vis3, "material")
                    n3 = rng.uniform(-0.04, 0.04)
                    ET.SubElement(m3, "ambient").text = f"{cr_rgb + n3:.2f} {cg_rgb + n3:.2f} {cb_rgb + n3:.2f} 1"
                    ET.SubElement(m3, "diffuse").text = f"{cr_rgb + n3:.2f} {cg_rgb + n3:.2f} {cb_rgb + n3:.2f} 1"


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


REQUIRED_STATS_KEYS = {
    "people", "civilian_people", "military_people",
    "vehicles", "civilian_vehicles", "concealed_vehicles",
    "vegetation", "concealment_distribution", "concealment_zones",
    "rejected_positions", "fallback_positions",
    "outside_bounds_rejections", "clipped_road_segments",
    "failed_shed_placements",
}


def _validate_stats(stats: dict) -> None:
    for key in REQUIRED_STATS_KEYS:
        if key not in stats:
            stats[key] = 0


def generate_rural_world_sdf(
    terrain_name: str,
    density: str,
    seed: int,
    exclusion_zones: list[tuple[float, float, float]] | None = None,
) -> tuple[str, dict]:
    """Generate complete rural ISR SDF world."""
    _reset_names()
    terrain = load_terrain(terrain_name)
    profile = RURAL_PROFILES[density]

    world_name = f"{terrain_name}_isr_rural_{density}"
    sdf, world = _sdf_header(world_name, profile["shadows"])

    # Terrain + helipad
    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "uri").text = f"model://{terrain_name}"
    inc2 = ET.SubElement(world, "include")
    ET.SubElement(inc2, "uri").text = "model://helipad"

    # Generate placements
    people, vehicles, roads, tree_lines = generate_rural_placements(
        terrain, density, seed, exclusion_zones,
    )

    # Generate base vegetation clusters
    veg_placements, _, _, clusters = generate_placements(
        terrain, density, seed, exclusion_zones,
    )

    rng = random.Random(seed)

    if density == "realistic_lite":
        new_clusters = []
        for cx, cy, cr in clusters:
            if not (180.0 <= cx <= 260.0 and -380.0 <= cy <= -320.0):
                cx = rng.uniform(190.0, 250.0)
                cy = rng.uniform(-370.0, -330.0)
            new_clusters.append((cx, cy, cr))
        clusters = new_clusters

    # Add dirt roads
    for i, road_points in enumerate(roads):
        _add_dirt_road(world, terrain, road_points, i, random.Random(seed + i))

    # Add sheds
    ez = exclusion_zones or [(0.0, 0.0, 15.0)]
    shed_idx = 0
    for _ in range(profile["sheds"]):
        if terrain.pixels:
            sx, sy = _flat_pos(terrain, rng, ez, density=density)
        else:
            sx, sy = _random_pos(terrain, rng, ez, density=density)
        if not _in_bounds(sx, sy, terrain):
            PLACEMENT_DIAGNOSTICS["failed_shed_placements"] += 1
            continue
        _add_shed(world, terrain, sx, sy, rng.uniform(0, 6.28), shed_idx, rng)
        shed_idx += 1

    # Add base vegetation (clamp to bounds — cluster offsets can escape)
    veg_idx = 0
    for p in veg_placements:
        if not _in_bounds(p.x, p.y, terrain):
            PLACEMENT_DIAGNOSTICS["outside_bounds_rejections"] += 1
            continue
        _add_inline_vegetation(world, veg_idx, p)
        veg_idx += 1

    # Add tree line vegetation
    _add_tree_line_vegetation(world, terrain, tree_lines, rng)

    # Add people
    for i, pp in enumerate(people):
        if pp.placement_type == "walking":
            _add_walking_actor(world, terrain, pp, random.Random(seed + i + 1000))
        else:
            _add_standing_person(world, terrain, pp)

    # Add vehicles
    for vp in vehicles:
        _add_fuel_vehicle_include(world, terrain, vp)

    # Add concealment zone markers
    _add_concealment_markers(world, clusters, terrain)

    ET.indent(sdf, space="  ")
    xml_str = '<?xml version="1.0" ?>\n' + ET.tostring(sdf, encoding="unicode")

    # Validate unique names
    dups = validate_unique_sdf_names(xml_str)
    if dups:
        raise RuntimeError(
            f"Duplicate SDF names generated ({len(dups)}):\n"
            + "\n".join(dups[:10])
        )

    stats = {
        "people": len(people),
        "civilian_people": sum(1 for p in people if p.target_type == "civilian"),
        "military_people": sum(1 for p in people if p.target_type == "military_like"),
        "vehicles": len(vehicles),
        "civilian_vehicles": sum(1 for v in vehicles if v.target_type == "civilian_vehicle"),
        "concealed_vehicles": sum(1 for v in vehicles if v.target_type == "concealed_vehicle"),
        "tree_lines": len(tree_lines),
        "vegetation": len(veg_placements),
        "roads": len(roads),
        "sheds": profile["sheds"],
        "clusters": len(clusters),
        "rejected_positions": PLACEMENT_DIAGNOSTICS.get("rejections_exclusion", 0)
                             + PLACEMENT_DIAGNOSTICS.get("rejections_slope", 0),
        "fallback_positions": PLACEMENT_DIAGNOSTICS.get("fallback_used", 0),
        "outside_bounds_rejections": PLACEMENT_DIAGNOSTICS.get("outside_bounds_rejections", 0),
        "clipped_road_segments": PLACEMENT_DIAGNOSTICS.get("clipped_road_segments", 0),
        "failed_shed_placements": PLACEMENT_DIAGNOSTICS.get("failed_shed_placements", 0),
        "concealment_distribution": {
            "open": sum(1 for p in people if p.concealment_level < 0.3),
            "partial": sum(1 for p in people if 0.3 <= p.concealment_level < 0.7),
            "high": sum(1 for p in people if p.concealment_level >= 0.7),
        },
    }

    _validate_stats(stats)

    return xml_str, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place rural ISR scene on real-terrain Gazebo world",
    )
    parser.add_argument("--world", required=True, help="World/model name (e.g. 1779343687303)")
    parser.add_argument("--density", choices=["light", "medium", "heavy", "camera_lite", "realistic_lite"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Generate all three densities")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Verify Fuel assets
    registry = verify_isr_all()
    missing = [k for k, v in registry.items() if not v.cache_dir or not v.cache_dir.exists()]
    if missing:
        print(f"WARNING: {len(missing)} Fuel models not cached (will use available ones):")
        for m in missing:
            print(f"  - {m}")
        print("  Run: python -m apps.tools.import_fuel_isr_assets")

    densities = ["light", "medium", "heavy"] if args.all else [args.density]
    out_dir = Path(args.output_dir) if args.output_dir else GZ_WORLDS_DIR

    for density in densities:
        print(f"\nGenerating {args.world}_isr_rural_{density}.sdf ...")
        xml_str, stats = generate_rural_world_sdf(args.world, density, args.seed)

        out_path = out_dir / f"{args.world}_isr_rural_{density}.sdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml_str)

        n_veg = xml_str.count('<model name="veg_')
        n_treeline_veg = xml_str.count('<model name="treeline_veg_')
        n_actors = xml_str.count("<actor")
        n_vehicles = xml_str.count('<name>vehicle_')
        n_markers = xml_str.count('<model name="marker_')
        n_roads = xml_str.count('<model name="road_')
        n_sheds = xml_str.count('<model name="shed_')

        print(f"  Written: {out_path}")
        print(f"  People: {stats['people']} (civ: {stats['civilian_people']}, mil: {stats['military_people']})")
        print(f"  Vehicles: {stats['vehicles']} (civ: {stats['civilian_vehicles']}, concealed: {stats['concealed_vehicles']})")
        print(f"  Vegetation: {n_veg} base + {n_treeline_veg} treeline")
        print(f"  Tree lines: {stats['tree_lines']}")
        print(f"  Roads: {n_roads} segments")
        print(f"  Sheds: {n_sheds}")
        print(f"  Actors: {n_actors}")
        print(f"  Markers: {n_markers}")
        cd = stats["concealment_distribution"]
        print(f"  Concealment: open={cd['open']}, partial={cd['partial']}, high={cd['high']}")
        total_entities = n_veg + n_treeline_veg + n_actors + n_vehicles + n_roads + n_sheds
        print(f"  Total entities: {total_entities}")
        est_fps = max(5, 60 - total_entities * 0.15)
        print(f"  Estimated FPS: ~{est_fps:.0f} (RTX 4090)")


if __name__ == "__main__":
    main()
