"""Download, verify, and catalog Gazebo Fuel assets for realistic scenes.

Usage:
    python -m apps.tools.import_fuel_assets
    python -m apps.tools.import_fuel_assets --verify-only
    python -m apps.tools.import_fuel_assets --strip-sensors
    python -m apps.tools.import_fuel_assets --report
"""

from __future__ import annotations

import argparse
import re
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
    category: str  # "vehicle_static" | "person_actor" | "person_static_mesh"
    z_offset_ground: float = 0.0
    mesh_yaw_internal: float = 0.0
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


# People models for realistic scenes
PEOPLE_MODELS: list[dict] = [
    {
        "owner": "OpenRobotics",
        "model_name": "Male visitor",
        "category": "person_actor",
        "z_offset_ground": 0.0,
        "skin_url": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male visitor/2/files/meshes/MaleVisitorWalk.dae",
    },
    {
        "owner": "Mingfei",
        "model_name": "actor",
        "category": "person_actor",
        "z_offset_ground": 1.0,
        "skin_url": "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae",
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
]

# Vehicle models for realistic scenes
VEHICLE_MODELS: list[dict] = [
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
]

# Opt-in vehicle — has sensors, only included if explicitly requested
SENSOR_VEHICLE_MODELS: list[dict] = [
    {
        "owner": "OpenRobotics",
        "model_name": "Prius Hybrid with sensors",
        "category": "vehicle_static",
        "z_offset_ground": 0.0,
        "semantic_name": "prius",
        "has_sensors": True,
    },
]

ALL_MODELS = PEOPLE_MODELS + VEHICLE_MODELS + SENSOR_VEHICLE_MODELS


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

    # Extract mesh URIs from visual/collision/skin/animation elements
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "uri":
            if elem.text:
                info.mesh_uris.append(elem.text.strip())
        elif tag == "filename":
            if elem.text and not elem.text.startswith("http"):
                info.mesh_uris.append(elem.text.strip())

    # Detect sensors
    for sensor in root.iter():
        tag = sensor.tag.split("}")[-1] if "}" in sensor.tag else sensor.tag
        if tag == "sensor":
            stype = sensor.get("type", sensor.get("name", "unknown"))
            if stype not in info.sensor_types:
                info.sensor_types.append(stype)
            info.has_sensors = True

    # Scan for material/texture files
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
            # Remote URI — check if file exists locally after Fuel download
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
    """Strip sensors/collisions from a model, write visual-only copy."""
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

    # Remove sensor elements and sensor links
    for parent in root.iter():
        to_remove = []
        for child in parent:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "sensor":
                to_remove.append(child)
            # Remove entire links named "sensors"
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

    # Copy model.config
    config_src = info.cache_dir / "model.config"
    if config_src.exists():
        shutil.copy2(config_src, out_path / "model.config")

    # Copy mesh and material directories
    for subdir in ("meshes", "materials"):
        src = info.cache_dir / subdir
        if src.exists() and src.is_dir():
            shutil.copytree(src, out_path / subdir, dirs_exist_ok=True)

    return out_path


def verify_all(models: list[dict] | None = None) -> dict[str, FuelModelInfo]:
    if models is None:
        models = ALL_MODELS
    registry = {}
    for m in models:
        info = scan_model_cache(m)
        key = f"{m['owner']}/{m['model_name']}"
        registry[key] = info
    return registry


def download_missing(models: list[dict] | None = None) -> list[str]:
    if models is None:
        models = PEOPLE_MODELS + VEHICLE_MODELS
    failures = []
    for m in models:
        result = _find_cached_version(m["owner"], m["model_name"])
        if result is None:
            if not download_model(m["owner"], m["model_name"]):
                failures.append(f"{m['owner']}/{m['model_name']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Fuel assets for realistic scenes")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--strip-sensors", action="store_true", help="Create visual-only copies of sensor models")
    parser.add_argument("--report", action="store_true", help="Print detailed model catalog")
    parser.add_argument("--include-prius", action="store_true", help="Include sensor-heavy Prius model")
    args = parser.parse_args()

    models = PEOPLE_MODELS + VEHICLE_MODELS
    if args.include_prius:
        models += SENSOR_VEHICLE_MODELS

    print(f"Fuel cache: {CACHE_ROOT}")
    print(f"Models: {len(models)}")
    print()

    registry = verify_all(models)

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
            print(f"FAIL: {len(missing_keys)} models missing")
            for k in missing_keys:
                print(f"  - {k}")
            sys.exit(1)
        print("All models cached.")
        return

    if not args.verify_only:
        failures = download_missing(models)
        if failures:
            print(f"FAIL: {len(failures)} downloads failed")
            sys.exit(1)

    if args.strip_sensors:
        assets_dir = PROJECT_ROOT / "ons-sentinel-core" / "configs" / "gz" / "assets" / "vehicles"
        for m in SENSOR_VEHICLE_MODELS:
            key = f"{m['owner']}/{m['model_name']}"
            info = registry.get(key)
            if info and info.has_sensors:
                out = create_visual_only_copy(info, assets_dir)
                if out:
                    print(f"Visual-only copy: {out}")
                else:
                    print(f"ERROR: could not create visual-only copy for {key}")

    print("Done.")


if __name__ == "__main__":
    main()
