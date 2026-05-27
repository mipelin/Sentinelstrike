#!/bin/bash
# Simulation-only autonomous Sentinel demo wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GZ_IP="${GZ_IP:-127.0.0.1}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/mipelin/.Xauthority}"

cd "${REPO_ROOT}"
exec python3 -m apps.tools.run_auto_sim_test "$@"
