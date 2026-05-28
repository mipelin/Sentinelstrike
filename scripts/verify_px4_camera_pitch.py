#!/usr/bin/env python3
"""Verify (and fix) the PX4 x500_mono_cam SDF camera pitch.

Upstream PX4 leaves the mono_cam at 0° pitch (horizontal). For ISR
search-and-follow the camera must look downward. This script ensures
the PX4 model is patched to +45° (0.785 rad) and is idempotent.

Convention:
  Gazebo/SDF  – positive pitch = nose/camera DOWN
  Sentinel cfg – pitch_deg negative = camera DOWN (e.g. -45.0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The pose strings we expect in the upstream model (horizontal)
OLD_INCLUDE_POSE = "<pose>.12 .03 .242 0 0 0</pose>"
OLD_JOINT_POSE = '<pose relative_to="base_link">.12 .03 .242 0 0 0</pose>'

# The corrected pose strings (45° down)
NEW_INCLUDE_POSE = "<pose>.12 .03 .242 0 0.785 0</pose>"
NEW_JOINT_POSE = '<pose relative_to="base_link">.12 .03 .242 0 0.785 0</pose>'

# Also handle variants with leading zeros
OLD_INCLUDE_POSE_ALT = "<pose>0.12 0.03 0.242 0 0 0</pose>"
OLD_JOINT_POSE_ALT = '<pose relative_to="base_link">0.12 0.03 0.242 0 0 0</pose>'
NEW_INCLUDE_POSE_ALT = "<pose>0.12 0.03 0.242 0 0.785 0</pose>"
NEW_JOINT_POSE_ALT = '<pose relative_to="base_link">0.12 0.03 0.242 0 0.785 0</pose>'

# Wrong poses that may have been introduced by experiments (UP instead of DOWN)
WRONG_INCLUDE_POSE = "<pose>.12 .03 .242 0 -0.785 0</pose>"
WRONG_JOINT_POSE = '<pose relative_to="base_link">.12 .03 .242 0 -0.785 0</pose>'
WRONG_INCLUDE_POSE_ALT = "<pose>0.12 0.03 0.242 0 -0.785 0</pose>"
WRONG_JOINT_POSE_ALT = '<pose relative_to="base_link">0.12 0.03 0.242 0 -0.785 0</pose>'


def patch_model(sdf_path: Path) -> bool:
    text = sdf_path.read_text()
    original = text

    # First fix any wrong (UP) poses
    wrong_replacements = [
        (WRONG_INCLUDE_POSE, NEW_INCLUDE_POSE),
        (WRONG_JOINT_POSE, NEW_JOINT_POSE),
        (WRONG_INCLUDE_POSE_ALT, NEW_INCLUDE_POSE_ALT),
        (WRONG_JOINT_POSE_ALT, NEW_JOINT_POSE_ALT),
    ]
    for old, new in wrong_replacements:
        text = text.replace(old, new)

    # Then fix upstream horizontal poses
    replacements = [
        (OLD_INCLUDE_POSE, NEW_INCLUDE_POSE),
        (OLD_JOINT_POSE, NEW_JOINT_POSE),
        (OLD_INCLUDE_POSE_ALT, NEW_INCLUDE_POSE_ALT),
        (OLD_JOINT_POSE_ALT, NEW_JOINT_POSE_ALT),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    if text == original:
        # Already correct (contains the exact correct poses)
        if NEW_INCLUDE_POSE in text or NEW_INCLUDE_POSE_ALT in text:
            return False  # no change needed / already patched
        print(f"WARNING: {sdf_path} does not contain expected horizontal or correct pose string.")
        print("  Please inspect the file manually.")
        return False

    sdf_path.write_text(text)
    print(f"PATCHED: {sdf_path}")
    print("  Camera pitch set to +0.785 rad (+45° down)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify/fix PX4 camera model pitch")
    parser.add_argument(
        "--px4-dir",
        type=Path,
        default=Path("/home/mipelin/projects/DRON/PX4-Autopilot"),
        help="Path to PX4-Autopilot repository",
    )
    args = parser.parse_args()

    model_sdf = (
        args.px4_dir
        / "Tools"
        / "simulation"
        / "gz"
        / "models"
        / "x500_mono_cam"
        / "model.sdf"
    )

    if not model_sdf.exists():
        print(f"ERROR: model file not found: {model_sdf}")
        return 1

    if patch_model(model_sdf):
        return 0

    print(f"OK: {model_sdf} already has correct camera pitch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
