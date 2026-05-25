#!/bin/bash
# Shared Gazebo rendering setup for headless camera sensor support.
#
# Sources into px4-headless.sh / px4-server-only.sh.
# Handles render engine selection, NVIDIA EGL, and diagnostics.
#
# Environment overrides:
#   GZ_RENDER_ENGINE=ogre|ogre2   Select rendering engine (default: ogre)
#   RENDER_ENGINE=ogre|ogre2      Alias for GZ_RENDER_ENGINE
#   HEADLESS_RENDERING=1          Force headless render path (default: 1)
#
# Provides:
#   GZ_RENDER_ENGINE     - resolved engine name (ogre or ogre2)
#   GZ_RENDER_ARGS       - CLI flags for gz sim
#   RENDER_DIAGNOSTICS() - print renderer diagnostics

# ── Resolve render engine ────────────────────────────────────────────

_GZ_RE="${GZ_RENDER_ENGINE:-${RENDER_ENGINE:-ogre}}"
case "${_GZ_RE}" in
    ogre|ogre2) ;;
    *)
        echo "WARNING: Unknown render engine '${_GZ_RE}', defaulting to ogre"
        _GZ_RE="ogre"
        ;;
esac
export GZ_RENDER_ENGINE="${_GZ_RE}"

# ── Detect GPU ───────────────────────────────────────────────────────

_HAS_NVIDIA=0
_NVIDIA_GPU=""
if command -v nvidia-smi &>/dev/null; then
    _NVIDIA_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    if [ -n "${_NVIDIA_GPU}" ]; then
        _HAS_NVIDIA=1
    fi
fi

# ── Set rendering environment ────────────────────────────────────────
#
# Key insight: OGRE2 + Mesa software EGL segfaults in headless mode.
# If NVIDIA GPU is present, use NVIDIA EGL — do NOT force software rendering.
# If no NVIDIA, fall back to software rendering with classic OGRE.

export HEADLESS="${HEADLESS:-1}"
unset DISPLAY

if [ "${_HAS_NVIDIA}" -eq 1 ]; then
    # NVIDIA EGL for headless rendering — no Mesa, no LIBGL_ALWAYS_SOFTWARE
    unset LIBGL_ALWAYS_SOFTWARE
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    # NVIDIA EGL device selection (use device 0 by default)
    : "${EGL_DEVICE_ID:=0}"
    export EGL_DEVICE_ID
else
    # No NVIDIA — software rendering with classic OGRE
    export LIBGL_ALWAYS_SOFTWARE=1
    # Force ogre when no GPU available — ogre2 + software Mesa segfaults
    if [ "${GZ_RENDER_ENGINE}" = "ogre2" ]; then
        echo "WARNING: No NVIDIA GPU detected — forcing ogre (ogre2 + Mesa segfaults)"
        export GZ_RENDER_ENGINE="ogre"
    fi
fi

# ── Build gz sim render arguments ────────────────────────────────────

GZ_RENDER_ARGS="--render-engine-server ${GZ_RENDER_ENGINE} --headless-rendering"
export GZ_RENDER_ARGS

# ── Diagnostics ──────────────────────────────────────────────────────

render_diagnostics() {
    echo "  === Rendering ==="
    echo "  Engine:           ${GZ_RENDER_ENGINE}"
    echo "  NVIDIA GPU:       $([ "${_HAS_NVIDIA}" -eq 1 ] && echo "${_NVIDIA_GPU}" || echo "(none)")"
    echo "  Headless:         ${HEADLESS}"
    echo "  Software render:  ${LIBGL_ALWAYS_SOFTWARE:-0}"
    echo "  GZ render args:   ${GZ_RENDER_ARGS}"
    echo ""

    # Probe EGL / OpenGL info
    echo "  === GPU Probe ==="
    if [ "${_HAS_NVIDIA}" -eq 1 ]; then
        _EGL_VENDOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
        echo "  EGL vendor:       NVIDIA (driver ${_EGL_VENDOR})"
        echo "  OpenGL vendor:    NVIDIA Corporation"
        echo "  GPU:              ${_NVIDIA_GPU}"
    elif command -v glxinfo &>/dev/null; then
        _GL_VENDOR=$(glxinfo -B 2>/dev/null | grep "OpenGL vendor" | cut -d: -f2 | xargs || echo "unknown")
        _GL_RENDERER=$(glxinfo -B 2>/dev/null | grep "OpenGL renderer" | cut -d: -f2 | xargs || echo "unknown")
        echo "  EGL vendor:       Mesa"
        echo "  OpenGL vendor:    ${_GL_VENDOR}"
        echo "  GPU:              ${_GL_RENDERER}"
    else
        echo "  EGL vendor:       (undetectable — install glxinfo)"
        echo "  OpenGL vendor:    (undetectable)"
        echo "  GPU:              (undetectable)"
    fi
    echo ""
}
