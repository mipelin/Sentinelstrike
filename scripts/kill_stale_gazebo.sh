#!/usr/bin/env bash
# Kill stale Gazebo/PX4 processes safely.
# Avoids pkill -f with broad patterns that match this script or the calling make.
set -euo pipefail

echo "Killing stale Gazebo/PX4 processes..."

# Exclude our own PID, parent PID, and any shell/make ancestor.
_self=$$
_excludes="grep|$$PPID|$_self"

# Build an exclusion regex from the current process tree up to make.
while read -r pid; do
    _excludes="$_excludes|$pid"
done < <(ps -o ppid= -p $$,PPID 2>/dev/null | tr -d ' ' | grep -v '^$$' || true)

# Kill PX4 SITL binary directly (no -f, matches exact process name).
pkill -9 -x px4 2>/dev/null || true

# Kill gz server/sim/gui by exact binary name.
pkill -9 -x "gz" 2>/dev/null || true
pkill -9 -x "ruby" 2>/dev/null || true

# Use pgrep+kill for any remaining gz-related processes, excluding self/parent.
remaining=$(pgrep -af "(build/px4_sitl_default/bin/px4|cmdsim)" 2>/dev/null \
    | grep -Ev "$_excludes" \
    | awk '{print $1}' || true)
if [ -n "$remaining" ]; then
    echo "$remaining" | xargs kill -9 2>/dev/null || true
fi

sleep 2

echo "Verifying clean state..."
if pgrep -x "gz" >/dev/null 2>&1; then
    echo "  WARNING: gz still running (try: pkill -9 -x gz)"
else
    echo "  gz: stopped"
fi
if pgrep -x "px4" >/dev/null 2>&1; then
    echo "  WARNING: px4 still running (try: pkill -9 -x px4)"
else
    echo "  px4: stopped"
fi

rm -rf /tmp/px4* 2>/dev/null || true
echo "Clean."
