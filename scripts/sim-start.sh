#!/bin/bash
# Start PX4 SITL + Gazebo with camera drone
set -euo pipefail

DISPLAY_NUM="${FLIGHTLAB_DISPLAY:-99}"
export DISPLAY=":${DISPLAY_NUM}"

PX4_DIR="${PX4_DIR:-/home/mipelin/projects/DRON/PX4-Autopilot}"
SENTINEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD="${WORLD:-${SENTINEL_DIR}/configs/gz/sentinel_camera.sdf}"
MODEL="${MODEL:-x500_mono_cam}"

echo "=== Starting PX4 + Gazebo Simulation ==="
echo "  Model:  ${MODEL}"
echo "  World:  ${WORLD}"
echo "  Display: ${DISPLAY}"
echo ""

# Check GPU memory
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE_VRAM" -lt 500 ]; then
    echo "ERROR: Only ${FREE_VRAM}MB GPU VRAM free. Stop llama-server first:"
    echo "  pkill -f llama-server"
    exit 1
fi

# Check Xvfb is running
if ! pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
    echo "Starting Xvfb..."
    Xvfb ":${DISPLAY_NUM}" -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
    sleep 1
fi

# Start Gazebo server + GUI in background
echo "Starting Gazebo..."
gz sim "${WORLD}" -r &
GZ_PID=$!
sleep 3

# Spawn the x500_mono_cam model
echo "Spawning ${MODEL}..."
gz sim -g  # This starts the GUI (needs display)

echo ""
echo "=== Gazebo Running (PID ${GZ_PID}) ==="
echo ""
echo "Now start PX4 SITL in another terminal:"
echo "  cd ${PX4_DIR}"
echo "  PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=${MODEL} make px4_sitl gz_${MODEL}"
echo ""
echo "Or use: make px4-sitl-start"
