#!/bin/bash
# stop_sentinel_stack.sh — Kill all Sentinel simulation processes cleanly.
set -euo pipefail

echo "=== Stopping Sentinel Stack ==="

tmux kill-session -t sentinel_sim 2>/dev/null && echo "  tmux session sentinel_sim killed" || true
pkill -9 -f "px4"            2>/dev/null && echo "  px4 killed"           || true
pkill -9 -f "gz sim"         2>/dev/null && echo "  gz sim killed"        || true
pkill -9 -f "ruby"           2>/dev/null && echo "  ruby killed"          || true
pkill -9 -f "mavsdk_server"  2>/dev/null && echo "  mavsdk_server killed" || true
pkill -9 -f "MicroXRCEAgent" 2>/dev/null && echo "  MicroXRCEAgent killed"|| true
pkill -9 -f "QGroundControl" 2>/dev/null && echo "  QGC killed"           || true

# Don't kill all python3 — only sentinel workers
pkill -9 -f "run_gazebo_yolo_test" 2>/dev/null && echo "  sentinel yolo test killed" || true
pkill -9 -f "run_gazebo_camera_probe" 2>/dev/null && echo "  sentinel probe killed" || true
pkill -9 -f "view_gazebo_camera" 2>/dev/null && echo "  sentinel viewer killed" || true
pkill -9 -f "run_gazebo_traffic" 2>/dev/null && echo "  sentinel traffic killed" || true

sleep 1

# Verify clean
set +o pipefail
REMAINING=$(ps aux | grep -E "[p]x4|[g]z sim|[Q]Ground" | wc -l)
set -o pipefail
if [ "$REMAINING" -gt 0 ]; then
    echo ""
    echo "WARNING: Some processes still running:"
    ps aux | grep -E "[p]x4|[g]z sim|[Q]Ground" | head -5
    echo "Run 'pkill -9 -f gz; pkill -9 px4' manually."
else
    echo ""
    echo "All clean."
fi

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader 2>/dev/null || true
