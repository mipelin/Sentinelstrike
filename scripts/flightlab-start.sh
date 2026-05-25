#!/bin/bash
# Start the virtual display + Sunshine for remote streaming
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPLAY_NUM=99

# --- Kill llama-server to free GPU VRAM ---
if pgrep -f "llama-server" >/dev/null 2>&1; then
    echo "Stopping llama-server to free GPU VRAM..."
    pkill -f "llama-server" 2>/dev/null || true
    sleep 2
    echo "GPU VRAM freed:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader
fi

# --- Start Xvfb virtual display ---
if ! pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
    echo "Starting Xvfb on :${DISPLAY_NUM} (1920x1080x24)..."
    Xvfb ":${DISPLAY_NUM}" -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
    sleep 1
    echo "Xvfb started (PID: $(pgrep -f "Xvfb :${DISPLAY_NUM}" | head -1))"
else
    echo "Xvfb already running on :${DISPLAY_NUM}"
fi

export DISPLAY=":${DISPLAY_NUM}"

# --- Verify NVIDIA GLX on virtual display ---
echo "Verifying GPU rendering on virtual display..."
if glxinfo 2>/dev/null | grep -q "NVIDIA"; then
    echo "  NVIDIA GLX: OK"
else
    echo "  WARNING: NVIDIA GLX not detected. Install nvidia GLX libs:"
    echo "    sudo apt install libgl1-nvidia-glx"
fi

echo ""
echo "=== Virtual Display Ready ==="
echo "  DISPLAY=:${DISPLAY_NUM}"
echo "  Resolution: 1920x1080"
echo ""
echo "To start Sunshine (remote streaming):"
echo "  DISPLAY=:${DISPLAY_NUM} sunshine"
echo ""
echo "To start Gazebo with rendering:"
echo "  DISPLAY=:${DISPLAY_NUM} gz sim -r"
echo ""
echo "To connect remotely:"
echo "  Install Moonlight on client -> connect to $(hostname -I | awk '{print $1}')"
