#!/bin/bash
# Stop flight lab services
set -euo pipefail

DISPLAY_NUM=99

echo "Stopping flight lab services..."

# Stop Sunshine
if pgrep sunshine >/dev/null 2>&1; then
    pkill sunshine 2>/dev/null || true
    echo "  Sunshine stopped"
fi

# Stop Gazebo
if pgrep "gz sim" >/dev/null 2>&1; then
    pkill -f "gz sim" 2>/dev/null || true
    echo "  Gazebo stopped"
fi

# Stop PX4 SITL
if pgrep px4 >/dev/null 2>&1; then
    pkill px4 2>/dev/null || true
    echo "  PX4 SITL stopped"
fi

# Stop QGroundControl
if pgrep QGroundControl >/dev/null 2>&1; then
    pkill QGroundControl 2>/dev/null || true
    echo "  QGroundControl stopped"
fi

# Stop Xvfb
if pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
    pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
    echo "  Xvfb stopped"
fi

echo ""
echo "Flight lab stopped. GPU status:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader
