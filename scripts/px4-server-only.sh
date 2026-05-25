#!/bin/bash
# Launch PX4 SITL + Gazebo server-only for real-terrain worlds.
#
# TRUE server-only: no GUI, no GLX, no X11, no DISPLAY.
# Camera topics publish. Sensors work. PX4 can arm.
#
# Usage:
#   WORLD=1779343687303 DENSITY=light ./scripts/px4-server-only.sh
#   WORLD=1779343687303 DENSITY=heavy ./scripts/px4-server-only.sh
#   ./scripts/px4-server-only.sh  # defaults: WORLD=1779343687303 DENSITY=light
#   GZ_RENDER_ENGINE=ogre ./scripts/px4-server-only.sh
#
# Makefile:
#   make px4-server-only-realterrain-light   WORLD=1779343687303
#   make px4-server-only-realterrain-medium  WORLD=1779343687303
#   make px4-server-only-realterrain-heavy   WORLD=1779343687303
#   make px4-server-only-realterrain-light   WORLD=1779343687303 RENDER_ENGINE=ogre
#
# Environment:
#   GZ_RENDER_ENGINE=ogre|ogre2  Rendering engine (default: ogre)
#   RENDER_ENGINE=ogre|ogre2     Alias for GZ_RENDER_ENGINE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PX4_DIR="${PX4_DIR:-/home/mipelin/projects/DRON/PX4-Autopilot}"

WORLD="${WORLD:-1779343687303}"
DENSITY="${DENSITY:-light}"

echo "=== PX4 Server-Only (Real Terrain) ==="
echo "  World:   ${WORLD}"
echo "  Density: ${DENSITY}"
echo "  PX4:     ${PX4_DIR}"
echo ""

# ── Kill stale ─────────────────────────────────────────────────────────

echo "[1/5] Killing stale processes..."
bash "${SCRIPT_DIR}/kill_stale_gazebo.sh" 2>/dev/null || true
sleep 1
echo ""

# ── Environment ────────────────────────────────────────────────────────

echo "[2/5] Setting up environment..."

GZ_ENV_SH="${PX4_DIR}/build/px4_sitl_default/rootfs/gz_env.sh"
if [ ! -f "${GZ_ENV_SH}" ]; then
    echo "ERROR: Build PX4 first: cd ${PX4_DIR} && make px4_sitl"
    exit 1
fi

# shellcheck source=/dev/null
source "${GZ_ENV_SH}"

# Add sentinel world paths
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}:${PROJECT_ROOT}/configs/gz"

# Set up rendering (engine selection, NVIDIA EGL, diagnostics)
# shellcheck source=gz-render-setup.sh
source "${SCRIPT_DIR}/gz-render-setup.sh"

echo "  PX4_GZ_MODELS:    ${PX4_GZ_MODELS}"
echo "  PX4_GZ_WORLDS:    ${PX4_GZ_WORLDS}"
echo "  PX4_GZ_PLUGINS:   ${PX4_GZ_PLUGINS}"
echo "  GZ_SIM_RESOURCE:  ...:${PROJECT_ROOT}/configs/gz"
echo ""
render_diagnostics

# ── Resolve world ──────────────────────────────────────────────────────

echo "[3/5] Resolving world..."

WORLD_SDF=""
# Determine scene type: realistic (Fuel models) or veg (primitive geometry)
REALISTIC="${REALISTIC:-0}"
if [ "${REALISTIC}" = "1" ]; then
    SCENE_WORLD="${WORLD}_realistic_${DENSITY}"
else
    SCENE_WORLD="${WORLD}_veg_${DENSITY}"
fi

for candidate in \
    "${PROJECT_ROOT}/configs/gz/${SCENE_WORLD}.sdf" \
    "${PX4_GZ_WORLDS}/${SCENE_WORLD}.sdf" \
    "${PROJECT_ROOT}/configs/gz/${WORLD}.sdf" \
    "${PX4_GZ_WORLDS}/${WORLD}.sdf"; do
    if [ -f "${candidate}" ]; then
        WORLD_SDF="${candidate}"
        break
    fi
done

if [ -z "${WORLD_SDF}" ]; then
    echo "ERROR: World SDF not found."
    echo "  Tried:"
    echo "    ${PROJECT_ROOT}/configs/gz/${SCENE_WORLD}.sdf"
    echo "    ${PX4_GZ_WORLDS}/${SCENE_WORLD}.sdf"
    echo "    ${PROJECT_ROOT}/configs/gz/${WORLD}.sdf"
    echo "    ${PX4_GZ_WORLDS}/${WORLD}.sdf"
    echo ""
    echo "  Generate first:"
    echo "    make realterrain-vegetation-${DENSITY}"
    echo "    make realterrain-realistic-${DENSITY}"
    exit 1
fi

WORLD_BASENAME=$(basename "${WORLD_SDF}" .sdf)

echo "  SDF:  ${WORLD_SDF}"
echo "  Name: ${WORLD_BASENAME}"
echo ""

# ── Start gz server ────────────────────────────────────────────────────

echo "[4/5] Starting Gazebo server (no GUI, engine=${GZ_RENDER_ENGINE})..."
echo "  Command: gz sim -r -s ${GZ_RENDER_ARGS} ${WORLD_SDF}"

gz sim -r -s ${GZ_RENDER_ARGS} "${WORLD_SDF}" &
GZ_PID=$!
echo "  Server PID: ${GZ_PID}"
echo ""

# Wait for server ready
echo "  Waiting for world '${WORLD_BASENAME}'..."
for i in $(seq 1 30); do
    if gz service -i --service "/world/${WORLD_BASENAME}/scene/info" 2>&1 | grep -q "Service providers"; then
        echo "  World ready."
        break
    fi
    if [ "${i}" -eq 30 ]; then
        echo "ERROR: Gazebo server timed out."
        kill "${GZ_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Verify topics
TOPIC_COUNT=$(gz topic -l 2>/dev/null | wc -l)
echo "  Topics: ${TOPIC_COUNT}"
echo ""

# ── Start PX4 ──────────────────────────────────────────────────────────

echo "[5/5] Starting PX4 SITL..."
echo ""

# Export PX4 environment — rcS uses PX4_SIM_MODEL to find airframe file
# by filename pattern: {id}_gz_x500_mono_cam (file 4010_gz_x500_mono_cam).
# px4-rc.gzsim auto-detects the running gz world via `gz topic -l | grep clock`.
export PX4_SIM_MODEL=gz_x500_mono_cam
export PX4_GZ_WORLD="${WORLD_BASENAME}"
export GZ_IP=127.0.0.1

echo "  === Environment ==="
echo "  PX4_SIM_MODEL=${PX4_SIM_MODEL}"
echo "  PX4_GZ_WORLD=${PX4_GZ_WORLD}"
echo "  GZ_IP=${GZ_IP}"
echo "  HEADLESS=${HEADLESS}"
echo "  GZ_RENDER_ENGINE=${GZ_RENDER_ENGINE}"
echo "  DISPLAY=${DISPLAY:-<unset>}"
echo ""

cd "${PX4_DIR}/build/px4_sitl_default"

exec bin/px4 \
    "${PX4_DIR}/build/px4_sitl_default/rootfs" \
    -s etc/init.d-posix/rcS
