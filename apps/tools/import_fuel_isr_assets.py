"""Download, verify, and catalog Gazebo Fuel assets for rural ISR training scenes.

Extends the base asset set from import_fuel_assets.py with additional models
for rural/military-style ISR testing: more people variety, more vehicles.

Usage:
    python -m apps.tools.import_fuel_isr_assets
    python -m apps.tools.import_fuel_isr_assets --verify-only
    python -m apps.tools.import_fuel_isr_assets --strip-sensors
    python -m apps.tools.import_fuel_isr_assets --report
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

FUEL_BASE = "https://fuel.gazebosim.org/1.0"
CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class FuelModelInfo:
    owner: str
    model_name: str
    category: str
    z_offset_ground: float = 0.0
    has_sensors: bool = False
    sensor_types: list[str] = field(default_factory=list)
    mesh_uris: list[str] = field(default_factory=list)
    material_files: list[str] = field(default_factory=list)
    version: int = 0
    cache_dir: Path = field(default_factory=Path)
    fuel_url: str = ""

    @property
    def fuel_include_uri(self) -> str:
        return f"{FUEL_BASE}/{self.owner}/models/{self.model_name}"


# ---------------------------------------------------------------------------
# ISR-specific people models
# ---------------------------------------------------------------------------

ISR_PEOPLE_MODELS: list[dict] = [
    # Existing models reused
    {
        "owner": "OpenRobotics",
        "model_name": "Male visitor",
        "category": "person_actor",
        "z_offset_ground": 0.0,
        "skin_url": f"{FUEL_BASE}/OpenRobotics/models/Male visitor/2/files/meshes/MaleVisitorWalk.dae",
    },
    {
        "owner": "OpenRobotics",
        "model_name": "FemaleVisitor",
        "category": "person_static_mesh",
        "z_offset_ground": 0.0,
    },
    {
        "owner": "OpenRobotics",
        "model_name": "Walking person",
        "category": "person_static_mesh",
        "z_offset_ground": 0.02,
    },
    {
        "owner": "Mingfei",
        "model_name": "actor",
        "category": "person_actor",
        "z_offset_ground": 1.0,
        "skin_url": f"{FUEL_BASE}/Mingfei/models/actor/tip/files/meshes/walk.dae",
    },
    # Additional ISR models
    {
        "owner": "OpenRobotics",
        "model_name": "Rescue Randy",
        "category": "person_static_mesh",
        "z_offset_ground": 0.0,
    },
    {
        "owner": "OpenRobotics",
        "model_name": "MaleVisitorOnPhone",
        "category": "person_static_mesh",
        "z_offset_ground": 0.0,
    },
]

# Actor skins — rotate for visual variety
ISR_ACTOR_SKINS = [
    f"{FUEL_BASE}/Mingfei/models/actor/tip/files/meshes/walk.dae",
    f"{FUEL_BASE}/OpenRobotics/models/Male visitor/2/files/meshes/MaleVisitorWalk.dae",
]

# Static standing people
ISR_STATIC_PEOPLE = [
    {"key": "female_visitor", "uri": f"{FUEL_BASE}/OpenRobotics/models/FemaleVisitor", "z_off": 0.0},
    {"key": "walking_person", "uri": f"{FUEL_BASE}/OpenRobotics/models/Walking person", "z_off": 0.02},
    {"key": "rescue_randy", "uri": f"{FUEL_BASE}/OpenRobotics/models/Rescue Randy", "z_off": 0.0},
    {"key": "male_phone", "uri": f"{FUEL_BASE}/OpenRobotics/models/MaleVisitorOnPhone", "z_off": 0.0},
]


# ---------------------------------------------------------------------------
# ISR-specific vehicle models
# ---------------------------------------------------------------------------

ISR_VEHICLE_MODELS: list[dict] = [
    # Existing models reused
    {
        "owner": "OpenRobotics",
        "model_name": "Hatchback",
        "category": "vehicle_static",
        "z_offset_ground": 0.0,
        "semantic_name": "hatchback",
    },
    {
        "owner": "OpenRobotics",
        "model_name": "Pickup",
        "category": "vehicle_static",
        "z_offset_ground": 0.0,
        "semantic_name": "pickup",
    },
    {
        "owner": "OpenRobotics",
        "model_name": "TruckBox",
        "category": "vehicle_static",
        "z_offset_ground": 0.0,
        "semantic_name": "truckbox",
    },
    {
        "owner": "OpenRobotics",
        "model_name": "Bus",
        "category": "vehicle_static",
        "z_offset_ground": 0.0,
        "semantic_name": "bus",
    },
    # Additional ISR models
    {
        "owner": "OpenRobotics",
        "model_name": "Prius Hybrid with sensors",
        "category": "vehicle_static",
        "z_offset_ground": 0.0,
        "semantic_name": "prius",
        "has_sensors": True,
    },
]

ISR_FUEL_VEHICLES = [
    {"key": "hatchback", "uri": f"{FUEL_BASE}/OpenRobotics/models/Hatchback", "z_off": 0.0},
    {"key": "pickup", "uri": f"{FUEL_BASE}/OpenRobotics/models/Pickup", "z_off": 0.0},
    {"key": "truckbox", "uri": f"{FUEL_BASE}/OpenRobotics/models/TruckBox", "z_off": 0.0},
    {"key": "bus", "uri": f"{FUEL_BASE}/OpenRobotics/models/Bus", "z_off": 0.0},
    {"key": "prius", "uri": f"{FUEL_BASE}/OpenRobotics/models/Prius Hybrid with sensors", "z_off": 0.0},
]

ISR_ALL_MODELS = ISR_PEOPLE_MODELS + ISR_VEHICLE_MODELS


# ---------------------------------------------------------------------------
# Cache scanning (reuses pattern from import_fuel_assets.py)
# ---------------------------------------------------------------------------


def _model_cache_dir(owner: str, model_name: str) -> Path:
    return CACHE_ROOT / owner.lower() / "models" / model_name.lower()


def _find_cached_version(owner: str, model_name: str) -> tuple[Path, int] | None:
    d = _model_cache_dir(owner, model_name)
    if not d.exists():
        return None
    for ver_dir in sorted(d.iterdir(), reverse=True):
        if ver_dir.is_dir() and (ver_dir / "model.config").exists():
            version = int(ver_dir.name) if ver_dir.name.isdigit() else 0
            return ver_dir, version
    return None


def scan_model_cache(model_def: dict) -> FuelModelInfo:
    info = FuelModelInfo(
        owner=model_def["owner"],
        model_name=model_def["model_name"],
        category=model_def["category"],
        z_offset_ground=model_def.get("z_offset_ground", 0.0),
        has_sensors=model_def.get("has_sensors", False),
        fuel_url=f"{FUEL_BASE}/{model_def['owner']}/models/{model_def['model_name']}",
    )

    result = _find_cached_version(model_def["owner"], model_def["model_name"])
    if result is None:
        return info

    ver_dir, version = result
    info.version = version
    info.cache_dir = ver_dir

    sdf_path = ver_dir / "model.sdf"
    if sdf_path.exists():
        _parse_model_sdf(info, sdf_path, ver_dir)

    return info


def _parse_model_sdf(info: FuelModelInfo, sdf_path: Path, ver_dir: Path) -> None:
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
    except ET.ParseError:
        return

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "uri":
            if elem.text:
                info.mesh_uris.append(elem.text.strip())
        elif tag == "filename":
            if elem.text and not elem.text.startswith("http"):
                info.mesh_uris.append(elem.text.strip())

    for sensor in root.iter():
        tag = sensor.tag.split("}")[-1] if "}" in sensor.tag else sensor.tag
        if tag == "sensor":
            stype = sensor.get("type", sensor.get("name", "unknown"))
            if stype not in info.sensor_types:
                info.sensor_types.append(stype)
            info.has_sensors = True

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.mtl"):
        for f in ver_dir.rglob(ext):
            info.material_files.append(str(f.relative_to(ver_dir)))


def download_model(owner: str, model_name: str) -> bool:
    url = f"{FUEL_BASE}/{owner}/models/{model_name}"
    print(f"  Downloading: {owner}/{model_name}")
    try:
        result = subprocess.run(
            ["gz", "fuel", "download", "-u", url, "-j", "4"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr.strip()}")
            return False
        print("    OK")
        return True
    except FileNotFoundError:
        print("    ERROR: 'gz fuel' CLI not found.")
        return False
    except subprocess.TimeoutExpired:
        print("    ERROR: download timed out")
        return False


def report_missing_textures(info: FuelModelInfo) -> list[str]:
    if not info.cache_dir or not info.cache_dir.exists():
        return ["cache directory missing"]

    missing = []
    for uri in info.mesh_uris:
        if uri.startswith("http"):
            parts = uri.split("/files/")
            if len(parts) == 2:
                local = info.cache_dir / parts[1]
                if not local.exists():
                    missing.append(f"mesh: {parts[1]}")
        else:
            local = info.cache_dir / uri
            if not local.exists():
                missing.append(f"mesh: {uri}")
    return missing


def create_visual_only_copy(info: FuelModelInfo, output_dir: Path) -> Path | None:
    if not info.cache_dir or not info.cache_dir.exists():
        return None
    sdf_path = info.cache_dir / "model.sdf"
    if not sdf_path.exists():
        return None

    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    for parent in root.iter():
        to_remove = []
        for child in parent:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "sensor":
                to_remove.append(child)
            if tag == "link" and child.get("name", "").lower() == "sensors":
                to_remove.append(child)
            if tag == "joint" and "sensor" in child.get("name", "").lower():
                to_remove.append(child)
        for child in to_remove:
            parent.remove(child)

    out_name = f"{info.model_name.lower().replace(' ', '_')}_visual_only"
    out_path = output_dir / out_name
    out_path.mkdir(parents=True, exist_ok=True)

    ET.indent(root, space="  ")
    xml_str = '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode")
    (out_path / "model.sdf").write_text(xml_str)

    config_src = info.cache_dir / "model.config"
    if config_src.exists():
        shutil.copy2(config_src, out_path / "model.config")

    for subdir in ("meshes", "materials"):
        src = info.cache_dir / subdir
        if src.exists() and src.is_dir():
            shutil.copytree(src, out_path / subdir, dirs_exist_ok=True)

    return out_path


def verify_isr_all(models: list[dict] | None = None) -> dict[str, FuelModelInfo]:
    if models is None:
        models = ISR_ALL_MODELS
    registry = {}
    for m in models:
        info = scan_model_cache(m)
        key = f"{m['owner']}/{m['model_name']}"
        registry[key] = info
    return registry


def download_missing(models: list[dict] | None = None) -> list[str]:
    if models is None:
        models = ISR_PEOPLE_MODELS + ISR_VEHICLE_MODELS
    failures = []
    for m in models:
        result = _find_cached_version(m["owner"], m["model_name"])
        if result is None:
            if not download_model(m["owner"], m["model_name"]):
                failures.append(f"{m['owner']}/{m['model_name']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ISR-specific Fuel assets for rural training scenes")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--strip-sensors", action="store_true", help="Create visual-only copies of sensor models")
    parser.add_argument("--report", action="store_true", help="Print detailed model catalog")
    args = parser.parse_args()

    models = ISR_PEOPLE_MODELS + ISR_VEHICLE_MODELS

    print(f"Fuel cache: {CACHE_ROOT}")
    print(f"ISR models: {len(models)}")
    print()

    registry = verify_isr_all(models)

    for key, info in registry.items():
        cached = "cached" if info.cache_dir and info.cache_dir.exists() else "MISSING"
        sensors = f" [sensors: {','.join(info.sensor_types)}]" if info.sensor_types else ""
        print(f"  [{cached}] {key} (v{info.version}){sensors}")

    print()

    if args.report:
        for key, info in registry.items():
            if not info.cache_dir or not info.cache_dir.exists():
                continue
            print(f"--- {key} ---")
            print(f"  Category: {info.category}")
            print(f"  Z offset: {info.z_offset_ground}")
            print(f"  Mesh URIs: {len(info.mesh_uris)}")
            for uri in info.mesh_uris[:5]:
                print(f"    {uri[:80]}")
            if info.material_files:
                print(f"  Materials: {len(info.material_files)}")
            missing = report_missing_textures(info)
            if missing:
                print(f"  MISSING: {missing}")
            print()

    if args.verify_only:
        missing_keys = [k for k, v in registry.items() if not v.cache_dir or not v.cache_dir.exists()]
        if missing_keys:
            print(f"FAIL: {len(missing_keys)} models not cached:")
            for k in missing_keys:
                print(f"  - {k}")
            sys.exit(1)
        print("All ISR models cached.")
        sys.exit(0)

    failures = download_missing(models)
    if failures:
        print(f"\nFailed to download {len(failures)} models:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    if args.strip_sensors:
        sensor_models = [m for m in models if m.get("has_sensors")]
        output_dir = PROJECT_ROOT / "configs" / "gz" / "assets" / "vehicles"
        for m in sensor_models:
            info = scan_model_cache(m)
            if info.has_sensors:
                out = create_visual_only_copy(info, output_dir)
                if out:
                    print(f"  Visual-only: {out}")

    print("\nISR asset import complete.")


if __name__ == "__main__":
    main()
