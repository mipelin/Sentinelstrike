"""Download and generate lightweight vegetation assets for real-terrain worlds.

Uses Gazebo Fuel models where available, and creates ultra-lightweight
fallback geometry models (cylinders + spheres) as visual-only SDF.

Usage:
    python -m apps.tools.download_gazebo_vegetation_assets
    python -m apps.tools.download_gazebo_vegetation_assets --verify-only
    python -m apps.tools.download_gazebo_vegetation_assets --generate-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FUEL_BASE = "https://fuel.gazebosim.org/1.0"

ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "gz" / "assets" / "vegetation"

# Fuel vegetation models — low-poly candidates
FUEL_VEGETATION: list[tuple[str, str, str]] = [
    # (owner, model_name, category)
    ("OpenRobotics", "Tree", "tree"),
    ("OpenRobotics", "Pine Tree", "pine"),
    ("OpenRobotics", "Oak Tree", "oak"),
    ("OpenRobotics", "Bush", "bush"),
    ("OpenRobotics", "Rock", "rock"),
]

FUEL_CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"


# ---------------------------------------------------------------------------
# Fallback geometry models — visual-only, zero mesh dependencies
# ---------------------------------------------------------------------------

def _write_sdf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _generate_tree(model_dir: Path) -> None:
    """Low-poly tree: cylinder trunk + sphere canopy."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.config").write_text("""\
<?xml version="1.0"?>
<model>
  <name>veg_tree</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>auto-generated</name></author>
  <description>Low-poly tree (visual only)</description>
</model>
""")
    (model_dir / "model.sdf").write_text("""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="veg_tree">
    <static>true</static>
    <link name="link">
      <!-- Trunk -->
      <visual name="trunk">
        <geometry><cylinder><radius>0.15</radius><length>3.0</length></cylinder></geometry>
        <pose>0 0 1.5 0 0 0</pose>
        <material>
          <ambient>0.35 0.25 0.15 1</ambient>
          <diffuse>0.45 0.32 0.18 1</diffuse>
        </material>
      </visual>
      <!-- Canopy -->
      <visual name="canopy">
        <geometry><sphere><radius>2.0</radius></sphere></geometry>
        <pose>0 0 4.2 0 0 0</pose>
        <material>
          <ambient>0.15 0.40 0.12 1</ambient>
          <diffuse>0.20 0.50 0.15 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


def _generate_bush(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.config").write_text("""\
<?xml version="1.0"?>
<model>
  <name>veg_bush</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>auto-generated</name></author>
  <description>Low-poly bush (visual only)</description>
</model>
""")
    (model_dir / "model.sdf").write_text("""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="veg_bush">
    <static>true</static>
    <link name="link">
      <visual name="body">
        <geometry><sphere><radius>0.8</radius></sphere></geometry>
        <pose>0 0 0.7 0 0 0</pose>
        <material>
          <ambient>0.18 0.42 0.14 1</ambient>
          <diffuse>0.22 0.52 0.18 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


def _generate_shrub(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.config").write_text("""\
<?xml version="1.0"?>
<model>
  <name>veg_shrub</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>auto-generated</name></author>
  <description>Small shrub (visual only)</description>
</model>
""")
    (model_dir / "model.sdf").write_text("""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="veg_shrub">
    <static>true</static>
    <link name="link">
      <visual name="body">
        <geometry><cylinder><radius>0.5</radius><length>0.8</length></cylinder></geometry>
        <pose>0 0 0.4 0 0 0</pose>
        <material>
          <ambient>0.2 0.38 0.15 1</ambient>
          <diffuse>0.25 0.48 0.2 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


def _generate_palm(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.config").write_text("""\
<?xml version="1.0"?>
<model>
  <name>veg_palm</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>auto-generated</name></author>
  <description>Palm tree (visual only)</description>
</model>
""")
    (model_dir / "model.sdf").write_text("""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="veg_palm">
    <static>true</static>
    <link name="link">
      <!-- Trunk -->
      <visual name="trunk">
        <geometry><cylinder><radius>0.12</radius><length>5.0</length></cylinder></geometry>
        <pose>0 0 2.5 0 0 0</pose>
        <material>
          <ambient>0.45 0.38 0.25 1</ambient>
          <diffuse>0.55 0.45 0.30 1</diffuse>
        </material>
      </visual>
      <!-- Crown -->
      <visual name="crown">
        <geometry><cylinder><radius>2.5</radius><length>0.6</length></cylinder></geometry>
        <pose>0 0 5.3 0 0 0</pose>
        <material>
          <ambient>0.12 0.42 0.10 1</ambient>
          <diffuse>0.18 0.52 0.14 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


def _generate_grass(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.config").write_text("""\
<?xml version="1.0"?>
<model>
  <name>veg_grass</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>auto-generated</name></author>
  <description>Grass patch (visual only)</description>
</model>
""")
    (model_dir / "model.sdf").write_text("""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="veg_grass">
    <static>true</static>
    <link name="link">
      <visual name="body">
        <geometry><cylinder><radius>1.5</radius><length>0.3</length></cylinder></geometry>
        <pose>0 0 0.15 0 0 0</pose>
        <material>
          <ambient>0.25 0.45 0.15 1</ambient>
          <diffuse>0.30 0.55 0.20 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


def _generate_rock(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.config").write_text("""\
<?xml version="1.0"?>
<model>
  <name>veg_rock</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>auto-generated</name></author>
  <description>Rock (visual only)</description>
</model>
""")
    (model_dir / "model.sdf").write_text("""\
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="veg_rock">
    <static>true</static>
    <link name="link">
      <visual name="body">
        <geometry><sphere><radius>0.6</radius></sphere></geometry>
        <pose>0 0 0.35 0 0 0</pose>
        <material>
          <ambient>0.45 0.42 0.38 1</ambient>
          <diffuse>0.55 0.52 0.48 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


FALLBACK_GENERATORS = {
    "veg_tree": _generate_tree,
    "veg_bush": _generate_bush,
    "veg_shrub": _generate_shrub,
    "veg_palm": _generate_palm,
    "veg_grass": _generate_grass,
    "veg_rock": _generate_rock,
}


def generate_fallback_models() -> None:
    """Generate all fallback geometry vegetation models."""
    print(f"Generating fallback vegetation models in {ASSET_DIR} ...")
    for name, gen_func in FALLBACK_GENERATORS.items():
        model_dir = ASSET_DIR / name
        if (model_dir / "model.sdf").exists():
            print(f"  [exists] {name}")
            continue
        gen_func(model_dir)
        print(f"  [created] {name}")
    print("Fallback models ready.")


# ---------------------------------------------------------------------------
# Fuel download (same pattern as download_gazebo_fuel_assets.py)
# ---------------------------------------------------------------------------

def _fuel_cached(owner: str, model_name: str) -> bool:
    d = FUEL_CACHE_ROOT / owner.lower() / "models" / model_name.lower()
    if not d.exists():
        return False
    for ver_dir in sorted(d.iterdir()):
        if ver_dir.is_dir() and (ver_dir / "model.config").exists():
            return True
    return False


def _fuel_download(owner: str, model_name: str) -> bool:
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


def download_fuel_models() -> None:
    """Download vegetation models from Gazebo Fuel."""
    print(f"Checking {len(FUEL_VEGETATION)} Fuel vegetation models...")
    missing = []
    for owner, model_name, _ in FUEL_VEGETATION:
        cached = _fuel_cached(owner, model_name)
        print(f"  [{'cached' if cached else 'MISSING'}] {owner}/{model_name}")
        if not cached:
            missing.append((owner, model_name))

    if not missing:
        print("All Fuel models cached.")
        return

    print(f"\nDownloading {len(missing)} models...")
    failures = []
    for owner, model_name in missing:
        if not _fuel_download(owner, model_name):
            failures.append(f"{owner}/{model_name}")

    if failures:
        print(f"\nWARNING: {len(failures)} Fuel downloads failed (fallback models will be used):")
        for f in failures:
            print(f"  - {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download/generate vegetation assets")
    parser.add_argument("--verify-only", action="store_true", help="Check only, no download/generate")
    parser.add_argument("--generate-only", action="store_true", help="Only generate fallback models")
    parser.add_argument("--fuel-only", action="store_true", help="Only download Fuel models")
    args = parser.parse_args()

    if args.verify_only:
        missing = []
        for name in FALLBACK_GENERATORS:
            if not (ASSET_DIR / name / "model.sdf").exists():
                missing.append(name)
        for owner, model_name, _ in FUEL_VEGETATION:
            if not _fuel_cached(owner, model_name):
                missing.append(f"fuel:{owner}/{model_name}")
        if missing:
            print(f"Missing: {', '.join(missing)}")
            sys.exit(1)
        print("All vegetation assets ready.")
        return

    if not args.fuel_only:
        generate_fallback_models()

    if not args.generate_only:
        download_fuel_models()

    print("\nDone.")


if __name__ == "__main__":
    main()
