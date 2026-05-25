#!/bin/bash
# Start PX4 SITL in the correct configuration
set -euo pipefail

PX4_DIR="${PX4_DIR:-/home/mipelin/projects/DRON/PX4-Autopilot}"
MODEL="${MODEL:-x500_mono_cam}"

echo "=== Starting PX4 SITL ==="
echo "  Model: ${MODEL}"
echo ""

cd "${PX4_DIR}"

# Build first if needed
if [ ! -f "build/px4_sitl_default/bin/px4" ]; then
    echo "Building PX4 SITL (first time, may take a few minutes)..."
    make px4_sitl
fi

echo "Launching PX4 SITL with model ${MODEL}..."
PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL="${MODEL}" make px4_sitl "gz_${MODEL}"
