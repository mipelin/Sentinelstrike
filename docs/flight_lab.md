# Flight Lab — Remote Drone Simulation Environment

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Ubuntu Server                         │
│                  (RTX 4090, 24GB)                        │
│                                                          │
│  ┌─────────┐  ┌──────────┐  ┌───────────────┐          │
│  │  Xvfb   │  │ Sunshine │  │    Moonlight   │ ◄──── Client PC
│  │ :99     │──│ (NVENC)  │──│ (H.265 stream)│          │
│  │ 1920x   │  │ :47984   │  │               │          │
│  │ 1080    │  └──────────┘  └───────────────┘          │
│  └────┬────┘                                              │
│       │ DISPLAY=:99                                       │
│  ┌────┴──────────────────────────────────────────────┐   │
│  │                                                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │   │
│  │  │  Gazebo  │  │  PX4     │  │ QGroundCtrl  │   │   │
│  │  │ Sim 8.11 │  │  SITL    │  │ (AppImage)   │   │   │
│  │  │ Harmonic │  │          │  │              │   │   │
│  │  └────┬─────┘  └────┬─────┘  └──────────────┘   │   │
│  │       │              │                             │   │
│  │       │ UDP 14540    │                             │   │
│  │       │              │                             │   │
│  │  ┌────┴──────────────┴──────┐                     │   │
│  │  │      Sentinel             │                     │   │
│  │  │  gz.transport13 → numpy   │                     │   │
│  │  │  → YOLO → Tracking → TAK │                     │   │
│  │  └──────────────────────────┘                     │   │
│  │                                                    │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Networking

| Port | Protocol | Service | Direction |
|------|----------|---------|-----------|
| 47984-47990 | TCP | Sunshine | Client → Server |
| 47998-48000 | UDP | Sunshine (stream) | Server → Client |
| 14540 | UDP | PX4 MAVLink (instance 0) | Server ↔ Client |
| 14541 | UDP | PX4 MAVLink (instance 1) | Server ↔ Client |
| 14556-14557 | UDP | PX4 MAVLink (offboard) | Server ↔ Client |
| 8787 | TCP | Sentinel dashboard | Client → Server |
| 11345 | TCP | Gazebo transport | Internal |

No firewall is active by default. If you enable UFW, run `make flightlab-setup` to open the required ports.

## Launch Sequence

### One-time setup (requires sudo)

```bash
make flightlab-setup
```

This installs Xvfb, Sunshine, QGroundControl, creates the project virtualenv (`.venv/`), installs MAVSDK Python and project deps inside the venv, and configures firewall rules. No system Python packages are modified — everything lives in the venv. Safe to rerun.

### Daily workflow

**Terminal 1: Start virtual display + free GPU VRAM**
```bash
cd ons-sentinel-core
make flightlab-start
```
This stops llama-server (freeing ~24GB VRAM) and starts Xvfb on :99.

**Terminal 1 (same): Start Sunshine for remote streaming**
```bash
DISPLAY=:99 sunshine &
```
Then open Moonlight on your client PC and connect to the server's IP.

**Terminal 2: Start Gazebo**
```bash
cd ons-sentinel-core
make sim-start
```
Starts Gazebo with the sentinel_urban world + Sensors plugin for camera rendering.

**Terminal 3: Start PX4 SITL**
```bash
cd ons-sentinel-core
make px4-sitl-start
```
Launches PX4 with the x500_mono_cam model. Drone appears in Gazebo.

**Terminal 4 (via Moonlight): QGroundControl**
```bash
DISPLAY=:99 /opt/QGroundControl.AppImage
```
Or use `make qgc-connect`. QGC connects to PX4 on UDP 14540 automatically.

**Terminal 5 (optional): Sentinel live perception**
```bash
cd ons-sentinel-core
make sentinel-live
```
Starts the Sentinel realtime loop with live Gazebo camera → YOLO pipeline.

**Terminal 6 (optional): FPV viewer**
```bash
make fpv-view
```
Opens an OpenCV window showing the drone's camera feed.

### Shutdown
```bash
make flightlab-stop
```
Stops everything (Gazebo, PX4, Sunshine, Xvfb).

### Quick status check
```bash
make flightlab-status
```

## Remote Access

### Moonlight (recommended)

1. Install Moonlight on your client (Windows/Mac/Linux/Android/iOS)
2. On server: `make flightlab-start && DISPLAY=:99 sunshine &`
3. On client: Open Moonlight → Add server → Enter server IP → Connect
4. You get a full 1920x1080 desktop at ~10ms latency via NVENC H.265

### QGroundControl remotely

Two options:
1. **Through Moonlight**: Run QGC on the server, see it via Moonlight stream
2. **Native on client**: Install QGC on your laptop, connect to server IP:14540

### Sentinel dashboard

Access from any browser: `http://SERVER_IP:8787`

## Simulation Worlds

| File | Description |
|------|-------------|
| `configs/gz/sentinel_camera.sdf` | Flat ground, minimal objects, camera-ready |
| `configs/gz/sentinel_urban.sdf` | Buildings, walls, enclosed area — best for perception testing |

Change the world:
```bash
make sim-start SENTINEL_WORLD=configs/gz/sentinel_urban.sdf
```

## Drone Models

| Model | Description |
|-------|-------------|
| `x500` | Basic quadcopter, no camera |
| `x500_mono_cam` | x500 + forward-facing RGB camera (1280x960, 30Hz) |
| `x500_mono_cam_down` | x500 + downward-facing camera |

Change the model:
```bash
make px4-sitl-start SENTINEL_MODEL=x500_mono_cam
```

## Troubleshooting

### "Insufficient GPU VRAM"
```
ERROR: Insufficient GPU VRAM. Stop llama-server first:
  pkill -f llama-server
```
Solution: `pkill -f llama-server` then retry. Gazebo rendering needs GPU memory.

### Gazebo renders but camera is black
- Verify the Sensors plugin is loaded: the world SDF must include `gz-sim-sensors-system`
- Use `make gazebo-topic-list` to check if camera topics exist
- If no topics: you're using the wrong world file (use `sentinel_urban.sdf` or `sentinel_camera.sdf`)

### Moonlight can't connect
- Verify Sunshine is running: `pgrep sunshine`
- Check firewall: `sudo ufw status`
- Verify ports 47984-47990 are open
- Try: `DISPLAY=:99 sunshine --verbose` for logs

### PX4 won't connect to Gazebo
- Verify Gazebo is running first: `pgrep -f "gz sim"`
- Check PX4 build: `cd PX4-Autopilot && make px4_sitl`
- Check model name matches: `PX4_SIM_MODEL=x500_mono_cam`

### Sentinel shows "No camera topics found"
- Gazebo must be running with a camera model (x500_mono_cam)
- The world must have the Sensors plugin
- Run `make gazebo-topic-list` to see available topics
- Topic name depends on world name: `/world/<world_name>/model/...`

### Xvfb won't start or GLX fails
```bash
# Verify NVIDIA GLX
DISPLAY=:99 glxinfo | grep "OpenGL renderer"
# Should show "NVIDIA GeForce RTX 4090"
# If not: install missing GLX libs
sudo apt install libgl1-nvidia-glx
```

## Jetson Migration Path

When moving to real hardware:

| Simulation | Hardware |
|------------|----------|
| Xvfb + Sunshine | HDMI out or SSH |
| Gazebo + x500_mono_cam | Real camera (CSI/RTSP) |
| `gazebo_camera` source type | `rtsp` or `webcam` source type |
| `GazeboCameraBridge` | `ManagedVideoSource` (RTSP/webcam) |
| PX4 SITL | Pixhawk via serial/USB |
| QGroundControl via Moonlight | QGroundControl native on laptop |

The Sentinel pipeline (perception, tracking, geolocalization, TAK) is source-agnostic. Swap the video source config and everything else stays the same.
