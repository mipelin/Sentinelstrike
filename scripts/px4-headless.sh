#!/bin/bash
# Launch PX4 SITL + Gazebo in TRUE server-only mode.
#
# Two-step launch:
#   1. Start gz sim server manually (no GUI, no GLX, no DISPLAY needed)
#   2. Start PX4 SITL binary — it detects the running world, skips its own
#      gz launch, and just spawns the model + starts gz_bridge
#
# Sensors preserved: IMU, magnetometer, air_pressure, navsat, camera image.
# Camera topics continue to publish. PX4 can arm.
#
# Usage:
#   HEADLESS=1 ./scripts/px4-headless.sh [world] [density]
#   ./scripts/px4-headless.sh sentinel_street medium
#   ./scripts/px4-headless.sh sentinel_street heavy
#   GZ_RENDER_ENGINE=ogre ./scripts/px4-headless.sh
#   RENDER_ENGINE=ogre2   ./scripts/px4-headless.sh
#
# Environment:
#   HEADLESS=1                Force headless mode (default when using this script)
#   PX4_DIR                   PX4-Autopilot root (default: ~/projects/DRON/PX4-Autopilot)
#   GZ_RENDER_ENGINE=ogre|ogre2  Rendering engine (default: ogre)
#   RENDER_ENGINE=ogre|ogre2     Alias for GZ_RENDER_ENGINE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PX4_DIR="${PX4_DIR:-/home/mipelin/projects/DRON/PX4-Autopilot}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"

WORLD="${1:-sentinel_street}"
DENSITY="${2:-medium}"

echo "=== PX4 Server-Only Launch ==="
echo "  World:   ${WORLD}"
echo "  Density: ${DENSITY}"
echo "  PX4:     ${PX4_DIR}"
echo ""

# ── Step 0: Kill stale processes ──────────────────────────────────────

echo "[1/6] Killing stale PX4/Gazebo processes..."
bash "${SCRIPT_DIR}/kill_stale_gazebo.sh" 2>/dev/null || true
echo ""

# ── Step 1: Set up Gazebo environment ─────────────────────────────────

echo "[2/6] Setting up Gazebo environment..."

GZ_ENV_SH="${PX4_DIR}/build/px4_sitl_default/rootfs/gz_env.sh"
if [ ! -f "${GZ_ENV_SH}" ]; then
    echo "ERROR: gz_env.sh not found at ${GZ_ENV_SH}"
    echo "  Build PX4 first: cd ${PX4_DIR} && make px4_sitl"
    exit 1
fi

# Source PX4's Gazebo environment (models, worlds, plugins, resource paths)
# shellcheck source=/dev/null
source "${GZ_ENV_SH}"

# Add sentinel world directories to resource path
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}:${PROJECT_ROOT}/configs/gz"

# Set up rendering (engine selection, NVIDIA EGL, diagnostics)
# shellcheck source=gz-render-setup.sh
source "${SCRIPT_DIR}/gz-render-setup.sh"

echo "  PX4_GZ_MODELS:  ${PX4_GZ_MODELS}"
echo "  PX4_GZ_WORLDS:  ${PX4_GZ_WORLDS}"
echo "  PX4_GZ_PLUGINS: ${PX4_GZ_PLUGINS}"
echo "  GZ_SIM_RESOURCE_PATH (extra): ${PROJECT_ROOT}/configs/gz"
echo ""
render_diagnostics

# ── Step 2: Resolve world SDF ─────────────────────────────────────────

echo "[3/6] Resolving world SDF..."

# Try density-specific file first, then base, then PX4 worlds
# Support both _veg_ (primitives) and _realistic_ (Fuel models) naming
REALISTIC="${REALISTIC:-0}"
if [ "${REALISTIC}" = "1" ]; then
    DENSITY_FILE="${WORLD}_realistic_${DENSITY}"
else
    DENSITY_FILE="${WORLD}_${DENSITY}"
fi

WORLD_SDF=""
for candidate in \
    "${PROJECT_ROOT}/configs/gz/${DENSITY_FILE}.sdf" \
    "${PROJECT_ROOT}/configs/gz/${WORLD}_${DENSITY}.sdf" \
    "${PROJECT_ROOT}/configs/gz/${WORLD}.sdf" \
    "${PX4_GZ_WORLDS}/${WORLD}.sdf"; do
    if [ -f "${candidate}" ]; then
        WORLD_SDF="${candidate}"
        break
    fi
done

if [ -z "${WORLD_SDF}" ]; then
    echo "ERROR: World SDF not found for '${WORLD}' (density: ${DENSITY})"
    echo "  Tried:"
    echo "    ${PROJECT_ROOT}/configs/gz/${DENSITY_FILE}.sdf"
    echo "    ${PROJECT_ROOT}/configs/gz/${WORLD}_${DENSITY}.sdf"
    echo "    ${PROJECT_ROOT}/configs/gz/${WORLD}.sdf"
    echo "    ${PX4_GZ_WORLDS}/${WORLD}.sdf"
    exit 1
fi

# Extract world name from SDF filename (without extension)
WORLD_BASENAME=$(basename "${WORLD_SDF}" .sdf)

echo "  SDF:  ${WORLD_SDF}"
echo "  Name: ${WORLD_BASENAME}"
echo ""

# ── Step 3: Start Gazebo server ────────────────────────────────────────

echo "[4/6] Starting Gazebo server (no GUI, engine=${GZ_RENDER_ENGINE})..."
echo "  Command: gz sim -r -s ${GZ_RENDER_ARGS} ${WORLD_SDF}"
echo ""

# -r: run immediately
# -s: server only (no GUI)
# --render-engine-server: explicit engine (ogre or ogre2)
# --headless-rendering: EGL offscreen rendering for sensors
gz sim -r -s ${GZ_RENDER_ARGS} "${WORLD_SDF}" &
GZ_SERVER_PID=$!

echo "  Gazebo server PID: ${GZ_SERVER_PID}"
echo ""

# Wait for server to be ready
echo "  Waiting for Gazebo world '${WORLD_BASENAME}' ..."
READY=0
for i in $(seq 1 30); do
    if gz service -i --service "/world/${WORLD_BASENAME}/scene/info" 2>&1 | grep -q "Service providers"; then
        READY=1
        break
    fi
    sleep 1
    echo "  ... waiting (${i}/30)"
done

if [ "${READY}" -ne 1 ]; then
    echo "ERROR: Gazebo server failed to start within 30s"
    echo "  Check: gz topic -l"
    kill "${GZ_SERVER_PID}" 2>/dev/null || true
    exit 1
fi

echo "  Gazebo server is ready."
echo ""

# ── Step 4: Verify camera/sensor topics ────────────────────────────────

echo "[5/6] Verifying Gazebo topics..."
TOPICS=$(gz topic -l 2>/dev/null || echo "")
CLOCK_COUNT=$(echo "${TOPICS}" | grep -c "clock" || echo "0")
echo "  Topics found: $(echo "${TOPICS}" | wc -l)"
echo "  Clock: $([ "${CLOCK_COUNT}" -gt 0 ] && echo "OK" || echo "MISSING")"
echo ""

# ── Step 5: Start PX4 SITL ────────────────────────────────────────────

echo "[6/6] Starting PX4 SITL..."
echo ""

# Export PX4 environment — rcS uses PX4_SIM_MODEL to find airframe file
# by filename pattern: {id}_gz_x500_mono_cam (file 4010_gz_x500_mono_cam).
# px4-rc.gzsim auto-detects the running gz world via `gz topic -l | grep clock`.
export PX4_SIM_MODEL=gz_x500_mono_cam
export PX4_GZ_WORLD="${WORLD_BASENAME}"
export GZ_IP=127.0.0.1

cd "${PX4_DIR}/build/px4_sitl_default"

echo "  === Environment ==="
echo "  PX4_SIM_MODEL=${PX4_SIM_MODEL}"
echo "  PX4_GZ_WORLD=${PX4_GZ_WORLD}"
echo "  GZ_IP=${GZ_IP}"
echo "  HEADLESS=${HEADLESS}"
echo "  GZ_RENDER_ENGINE=${GZ_RENDER_ENGINE}"
echo "  DISPLAY=${DISPLAY:-<unset>}"
echo ""

exec bin/px4 \
    "${PX4_DIR}/build/px4_sitl_default/rootfs" \
    -s etc/init.d-posix/rcS
