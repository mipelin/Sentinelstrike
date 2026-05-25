#!/bin/bash
# Flight Lab Infrastructure Setup — run once with sudo
# Installs: Xvfb, Sunshine (remote streaming), build tools
set -euo pipefail

PROJECT_DIR="/home/mipelin/projects/DRON/ons-sentinel-core"
VENV_DIR="${PROJECT_DIR}/.venv"
ACTUAL_USER="$(logname 2>/dev/null || echo mipelin)"

echo "=== Flight Lab Setup ==="
echo ""
echo "  User:        ${ACTUAL_USER}"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Venv dir:    ${VENV_DIR}"
echo ""

# 1. Xvfb + X11 utils
echo "[1/6] Installing Xvfb and X11 utilities..."
apt-get update -qq
apt-get install -y xvfb x11-xserver-utils xdotool x11vnc python3-venv

# 2. Sunshine (remote streaming via NVENC)
echo "[2/6] Installing Sunshine..."
if ! command -v sunshine &>/dev/null; then
    SUNSHINE_DEB=$(mktemp /tmp/sunshine-XXXXXX.deb)
    echo "  Downloading Sunshine latest release..."
    curl -sL https://github.com/LizardByte/Sunshine/releases/latest/download/sunshine-ubuntu-24.04-amd64.deb -o "$SUNSHINE_DEB"
    apt-get install -y "$SUNSHINE_DEB"
    rm -f "$SUNSHINE_DEB"
else
    echo "  Sunshine already installed"
fi

# 3. QGroundControl
echo "[3/6] Installing QGroundControl..."
if [ ! -f /opt/QGroundControl.AppImage ]; then
    QGC_URL="https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage"
    curl -sL "$QGC_URL" -o /opt/QGroundControl.AppImage
    chmod +x /opt/QGroundControl.AppImage
    usermod -aG dialout "$ACTUAL_USER"
else
    echo "  QGroundControl already installed"
fi

# 4. Create project virtualenv if missing
echo "[4/6] Ensuring project virtualenv exists..."
if [ ! -f "${VENV_DIR}/bin/python" ]; then
    echo "  Creating virtualenv at ${VENV_DIR}..."
    runuser -u "$ACTUAL_USER" -- python3 -m venv "${VENV_DIR}"
else
    echo "  Virtualenv already exists at ${VENV_DIR}"
fi

echo "  Python: $(readlink -f "${VENV_DIR}/bin/python")"
echo "  Pip:    $("${VENV_DIR}/bin/python" -m pip --version)"

# 5. Install MAVSDK + project deps in venv (NOT system pip — PEP 668)
echo "[5/6] Installing MAVSDK Python in project virtualenv..."
runuser -u "$ACTUAL_USER" -- "${VENV_DIR}/bin/python" -m pip install --upgrade pip
runuser -u "$ACTUAL_USER" -- "${VENV_DIR}/bin/python" -m pip install mavsdk
# Install project with relevant extras if pyproject.toml exists
if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
    runuser -u "$ACTUAL_USER" -- "${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[mavlink]" 2>/dev/null \
        || runuser -u "$ACTUAL_USER" -- "${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}" 2>/dev/null \
        || true
fi
echo "  MAVSDK Python installed in venv."

# 6. Firewall rules for flight lab
echo "[6/6] Configuring network..."
for port in 47984 47985 47986 47987 47988 47989 47990 8787 11345; do
    ufw allow "$port/tcp" 2>/dev/null || true
done
for port in 47998 47999 48000 14540 14541 14556 14557; do
    ufw allow "$port/udp" 2>/dev/null || true
done

echo ""
echo "=== Setup Complete ==="
echo "  Virtualenv: ${VENV_DIR}"
echo "  Python:     $(readlink -f "${VENV_DIR}/bin/python")"
echo "  Run: make flightlab-start"
echo "  Then connect Moonlight to this machine's IP"
