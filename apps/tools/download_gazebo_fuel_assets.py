"""Download and verify Gazebo Fuel assets for the sentinel_street simulation.

Usage:
    python -m apps.tools.download_gazebo_fuel_assets
    python -m apps.tools.download_gazebo_fuel_assets --verify-only

Assets are cached in ~/.gz/fuel/ by the gz fuel CLI.
This script downloads models that are missing and verifies integrity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FUEL_BASE = "https://fuel.gazebosim.org/1.0"

# Models used by sentinel_street.sdf
REQUIRED_MODELS: list[tuple[str, str]] = [
    # (owner, model_name)
    # People
    ("OpenRobotics", "Male visitor"),
    ("OpenRobotics", "FemaleVisitor"),
    ("OpenRobotics", "Walking person"),
    ("Mingfei", "actor"),
    # Vehicles
    ("OpenRobotics", "Hatchback"),
    ("OpenRobotics", "Pickup"),
    ("OpenRobotics", "TruckBox"),
    ("OpenRobotics", "Bus"),
    ("OpenRobotics", "Prius Hybrid with sensors"),
]

# gz fuel caches at ~/.gz/fuel/fuel.gazebosim.org/<owner_lower>/models/<model_lower>/
CACHE_ROOT = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org"


def _model_cache_dir(owner: str, model_name: str) -> Path:
    return CACHE_ROOT / owner.lower() / "models" / model_name.lower()


def _is_cached(owner: str, model_name: str) -> bool:
    d = _model_cache_dir(owner, model_name)
    if not d.exists():
        return False
    # Check at least one version directory with model.config
    for ver_dir in sorted(d.iterdir()):
        if ver_dir.is_dir() and (ver_dir / "model.config").exists():
            return True
    return False


def _download(owner: str, model_name: str) -> bool:
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
        print(f"    OK")
        return True
    except FileNotFoundError:
        print(f"    ERROR: 'gz fuel' CLI not found. Install gz-tools or run manually:")
        print(f"      gz fuel download -u '{url}'")
        return False
    except subprocess.TimeoutExpired:
        print(f"    ERROR: download timed out")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gazebo Fuel assets")
    parser.add_argument("--verify-only", action="store_true", help="Check cache, don't download")
    args = parser.parse_args()

    print(f"Fuel asset cache: {CACHE_ROOT}")
    print(f"Required models: {len(REQUIRED_MODELS)}")
    print()

    missing = []
    for owner, model_name in REQUIRED_MODELS:
        cached = _is_cached(owner, model_name)
        status = "cached" if cached else "MISSING"
        print(f"  [{status}] {owner}/{model_name}")
        if not cached:
            missing.append((owner, model_name))

    print()

    if args.verify_only:
        if missing:
            print(f"FAIL: {len(missing)} models missing. Run without --verify-only to download.")
            sys.exit(1)
        print("All models cached.")
        return

    if not missing:
        print("All models already cached.")
        return

    print(f"Downloading {len(missing)} missing models...")
    print()

    failures = []
    for owner, model_name in missing:
        if not _download(owner, model_name):
            failures.append((owner, model_name))

    print()

    # Re-verify
    still_missing = []
    for owner, model_name in REQUIRED_MODELS:
        if not _is_cached(owner, model_name):
            still_missing.append(f"{owner}/{model_name}")

    if still_missing:
        print(f"FAIL: {len(still_missing)} models still missing:")
        for m in still_missing:
            print(f"  - {m}")
        print()
        print("Manual download commands:")
        for m in still_missing:
            parts = m.split("/")
            print(f"  gz fuel download -u '{FUEL_BASE}/{m}'")
        sys.exit(1)

    print(f"All {len(REQUIRED_MODELS)} models ready.")


if __name__ == "__main__":
    main()
