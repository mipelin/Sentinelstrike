#!/bin/bash
# start_sentinel_stack.sh — Launch PX4 SITL + Gazebo + QGC for Sentinel ISR simulation.
#
# Fixes applied:
#   1. GZ_IP=127.0.0.1       — Prevent gz transport binding to Docker bridge (172.17.0.1)
#   2. PX4_GZ_WORLD=default   — Default world works for camera rendering.
#                                ISR rural world (1779343687303_isr_rural_light) has a known
#                                Gazebo Harmonic rendering bug: sensors register topics but
#                                never produce frames due to texture budget / scene complexity.
#   3. DISPLAY=:0              — Use real X display for GPU rendering
#
# Usage:
#   ./start_sentinel_stack.sh                          # Default world + GUI + QGC
#   ./start_sentinel_stack.sh --isr-world              # ISR rural world (camera may not render)
#   ./start_sentinel_stack.sh --no-qgc                 # Skip QGC launch
#   ./start_sentinel_stack.sh --validate               # Run camera validation after startup
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
PX4_DIR="${PX4_DIR:-/home/mipelin/projects/DRON/PX4-Autopilot}"
SENTINEL_DIR="${SENTINEL_DIR:-/home/mipelin/projects/DRON/ons-sentinel-core}"
QGC_BIN="${QGC_BIN:-/home/mipelin/QGroundControl.AppImage}"
ISR_WORLD="1779343687303_isr_rural_light"
ISR_LITE_WORLD="1779343687303_isr_rural_camera_lite"
ISR_REALISTIC_LITE_WORLD="1779343687303_isr_rural_realistic_lite"
ISR_REALISTIC_LITE_V2_WORLD="1779343687303_isr_rural_realistic_lite_v2"
MODEL="${PX4_SIM_MODEL:-x500_mono_cam}"

# Default to the world that has working camera rendering
WORLD="default"

NO_QGC=0
VALIDATE=0
SPAWN_POSE=""
for arg in "$@"; do
    case "$arg" in
        --isr-world)           WORLD="${ISR_WORLD}" ;;
        --isr-lite)            WORLD="${ISR_LITE_WORLD}"
                               SPAWN_POSE="${PX4_GZ_MODEL_POSE:-220,-350,20,0,0,0}" ;;
        --isr-realistic-lite)  WORLD="${ISR_REALISTIC_LITE_WORLD}"
                               SPAWN_POSE="${PX4_GZ_MODEL_POSE:-220,-350,20,0,0,0}" ;;
        --isr-realistic-lite-v2) WORLD="${ISR_REALISTIC_LITE_V2_WORLD}"
                                 SPAWN_POSE="${PX4_GZ_MODEL_POSE:-220,-350,20,0,0,0}" ;;
        --no-qgc)              NO_QGC=1 ;;
        --validate)            VALIDATE=1 ;;
    esac
done

# ── Pre-flight checks ────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Sentinel Strike — Stack Launcher               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  PX4 dir:      ${PX4_DIR}"
echo "  Sentinel dir: ${SENTINEL_DIR}"
echo "  World:        ${WORLD}"
echo "  Model:        ${MODEL}"
echo ""
if [ "${WORLD}" = "${ISR_WORLD}" ]; then
    echo "  WARNING: ISR rural world has a known camera rendering bug in"
    echo "  Gazebo Harmonic. Camera topics register but frames don't publish."
    echo "  Use --validate to check, or omit --isr-world for default world."
    echo ""
elif [ "${WORLD}" = "${ISR_LITE_WORLD}" ]; then
    echo "  INFO: Using ISR Camera Lite world optimized for FPS and camera reliability."
    echo "  Drone will spawn at target pose: ${SPAWN_POSE}"
    echo ""
elif [ "${WORLD}" = "${ISR_REALISTIC_LITE_WORLD}" ]; then
    echo "  INFO: Using ISR Realistic Lite world with realistic targets and snapped assets."
    echo "  Drone will spawn at target pose: ${SPAWN_POSE}"
    echo ""
elif [ "${WORLD}" = "${ISR_REALISTIC_LITE_V2_WORLD}" ]; then
    echo "  INFO: Using ISR Realistic Lite V2 world with improved human visibility and grounded calibration groups."
    echo "  Drone will spawn at target pose: ${SPAWN_POSE}"
    echo ""
fi

# Check GPU
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "0")
if [ "${FREE_VRAM}" -lt 500 ]; then
    echo "ERROR: Only ${FREE_VRAM}MB GPU VRAM free. Free up GPU memory first."
    exit 1
fi
echo "  GPU VRAM free: ${FREE_VRAM} MB"

# Check PX4 build
if [ ! -f "${PX4_DIR}/build/px4_sitl_default/bin/px4" ]; then
    echo ""
    echo "PX4 not built. Building..."
    cd "${PX4_DIR}" && make px4_sitl
fi

# Kill stale processes
echo ""
echo "=== Cleaning stale processes ==="
bash "${SENTINEL_DIR}/scripts/stop_sentinel_stack.sh" 2>/dev/null || true
sleep 2

# ── Environment setup ────────────────────────────────────────────────
cd "${PX4_DIR}"

# Source PX4 Gazebo environment (sets GZ_SIM_RESOURCE_PATH, plugin paths, etc.)
set +u
source build/px4_sitl_default/rootfs/gz_env.sh
set -u

# ── FIX 1: Force gz transport to localhost (avoid Docker bridge) ────
export GZ_IP=127.0.0.1

# ── FIX 2: Display settings ─────────────────────────────────────────
unset HEADLESS
export DISPLAY=:0
export XAUTHORITY=/home/mipelin/.Xauthority

# ── FIX 3: World selection ──────────────────────────────────────────
export PX4_GZ_WORLD="${WORLD}"

# ── Launch PX4 + Gazebo inside tmux ──────────────────────────────────
echo ""
echo "=== Launching PX4 SITL + Gazebo inside tmux (session: sentinel_sim) ==="
echo "  World:   ${WORLD}"
echo "  Model:   gz_${MODEL}"
echo "  GZ_IP:   ${GZ_IP}"
echo "  DISPLAY: ${DISPLAY}"
echo ""

# Start detached tmux session
tmux new-session -d -s sentinel_sim -n "PX4_Gazebo" || true

# Send setup and launch commands to the tmux session
tmux send-keys -t sentinel_sim:PX4_Gazebo "cd ${PX4_DIR}" C-m
tmux send-keys -t sentinel_sim:PX4_Gazebo "source build/px4_sitl_default/rootfs/gz_env.sh" C-m
tmux send-keys -t sentinel_sim:PX4_Gazebo "export GZ_IP=127.0.0.1" C-m
tmux send-keys -t sentinel_sim:PX4_Gazebo "export DISPLAY=:0" C-m
tmux send-keys -t sentinel_sim:PX4_Gazebo "export XAUTHORITY=/home/mipelin/.Xauthority" C-m
if [ "${WORLD}" = "default" ]; then
    tmux send-keys -t sentinel_sim:PX4_Gazebo "unset PX4_GZ_WORLD" C-m
else
    tmux send-keys -t sentinel_sim:PX4_Gazebo "export PX4_GZ_WORLD=${WORLD}" C-m
fi
if [ -n "${SPAWN_POSE}" ]; then
    tmux send-keys -t sentinel_sim:PX4_Gazebo "export PX4_GZ_MODEL_POSE=${SPAWN_POSE}" C-m
else
    tmux send-keys -t sentinel_sim:PX4_Gazebo "unset PX4_GZ_MODEL_POSE" C-m
fi
tmux send-keys -t sentinel_sim:PX4_Gazebo "make px4_sitl gz_${MODEL}" C-m

# ── Wait for PX4 to initialize ──────────────────────────────────────
echo "Waiting for PX4 to initialize (35s)..."
sleep 35

# ── Check if PX4 is running ─────────────────────────────────────────
if ! pgrep -f "px4" >/dev/null 2>&1; then
    echo "ERROR: PX4 failed to start. Check the terminal output."
    exit 1
fi

echo "PX4 is running."

# ── Launch QGroundControl ────────────────────────────────────────────
if [ "${NO_QGC}" -eq 0 ]; then
    echo ""
    echo "=== Launching QGroundControl ==="
    if [ -x "${QGC_BIN}" ]; then
        nohup "${QGC_BIN}" >/dev/null 2>&1 &
        echo "  QGC started (PID $!)."
        echo "  QGC should auto-connect on UDP 14550 within a few seconds."
        echo "  If not: Settings > Comm Links > Add > UDP > Port 14550 > Connect"
    else
        echo "  QGC not found at ${QGC_BIN}. Launch manually."
    fi
fi

# ── Status report ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Sentinel Stack — Status                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check UDP ports
echo "  MAVLink ports:"
ss -lunp 2>/dev/null | grep -E "14550|18570|14540|14580" | while read line; do
    echo "    ${line}"
done

# Check camera topics
echo ""
echo "  Camera topics:"
gz topic -l 2>/dev/null | grep -E "camera/image" | while read line; do
    echo "    ${line}"
done

# ── Validation ───────────────────────────────────────────────────────
CAMERA_TOPIC="/world/${WORLD}/model/${MODEL}_0/link/camera_link/sensor/camera/image"

if [ "${VALIDATE}" -eq 1 ]; then
    echo ""
    echo "=== Running camera validation ==="
    echo "  Testing topic: ${CAMERA_TOPIC}"

    CAMERA_SAMPLE=$(timeout 5 gz topic -e -t "${CAMERA_TOPIC}" 2>&1 | sed -n '1,12p' || true)
    if printf "%s\n" "${CAMERA_SAMPLE}" | grep -q "header"; then
        echo "  CAMERA OK: frames are being published!"
    else
        echo "  CAMERA WARNING: no frames on expected topic."
        ALT_TOPIC=$(gz topic -l 2>/dev/null | grep "camera/image" | head -1)
        if [ -n "${ALT_TOPIC}" ]; then
            echo "  Found alternative topic: ${ALT_TOPIC}"
            ALT_CAMERA_SAMPLE=$(timeout 5 gz topic -e -t "${ALT_TOPIC}" 2>&1 | sed -n '1,12p' || true)
            if printf "%s\n" "${ALT_CAMERA_SAMPLE}" | grep -q "header"; then
                echo "  CAMERA OK on alternative topic!"
                CAMERA_TOPIC="${ALT_TOPIC}"
            else
                echo "  CAMERA FAIL: topic exists but no frames."
                echo "  This is a known Gazebo Harmonic rendering bug with complex worlds."
                echo "  Fix: use default world (omit --isr-world)."
            fi
        else
            echo "  CAMERA FAIL: no camera topics found at all."
            echo "  Check that the model was spawned correctly."
        fi
    fi
fi

echo ""
echo "=== Startup complete ==="
echo ""
echo "Camera topic:"
echo "  ${CAMERA_TOPIC}"
echo ""
echo "Next steps:"
echo "  1. Verify QGC connection (should auto-connect on UDP 14550)"
echo "  2. Test camera (mock backend):"
echo "     cd ${SENTINEL_DIR} && source .venv/bin/activate"
echo "     python3 -m apps.tools.run_gazebo_yolo_test \\"
echo "       --topic '${CAMERA_TOPIC}' \\"
echo "       --backend mock --max-frames 100"
echo "  3. Run with YOLO CUDA:"
echo "     python3 -m apps.tools.run_gazebo_yolo_test \\"
echo "       --topic '${CAMERA_TOPIC}' \\"
echo "       --backend yolo --target-fps 25 --max-frames 300 --profile"
echo "  4. Full ISR mode:"
echo "     python3 -m apps.tools.run_gazebo_yolo_test \\"
echo "       --topic '${CAMERA_TOPIC}' \\"
echo "       --backend yolo --track isr --target-fps 25 --max-frames 300 \\"
echo "       --use-workers --follow-mode standoff --standoff-distance 10 \\"
echo "       --standoff-altitude 10 --debug-target-selection \\"
echo "       --debug-world-projection --profile"
echo ""
