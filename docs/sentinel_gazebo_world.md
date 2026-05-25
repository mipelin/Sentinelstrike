# Sentinel Street — Gazebo World

Urban street environment for Sentinel Strike with buildings, cars, people, traffic signs, and obstacles. Designed for the `x500_mono_cam` drone to fly at 10-50m altitude and capture images suitable for YOLO detection.

## What's in the World

| Element | Count | Description |
|---------|-------|-------------|
| Roads | 2 | Main east-west road + cross road with lane markings and crosswalks |
| Sidewalks | 2 | North and south sides |
| Buildings | 8 | Office towers, warehouse, residential blocks, shop, garage, apartment |
| Cars | 5 | Red sedan, blue SUV, white van, black sedan, silver car |
| People | 6 | Mannequins on sidewalks and near crosswalk |
| Obstacles | 5 | Traffic barriers, dumpster, jersey barrier, trash can |
| Signs | 2 | Stop sign, traffic light |
| Street lamps | 2 | With arms and lamp heads |
| Trees | 3 | Trunk + canopy spheres |

All objects are SDF primitives (boxes, cylinders, spheres) — no external assets needed.

## Architecture

The SDF has **no inline plugins** — matching PX4's `default.sdf`. All Gazebo system plugins (Physics, Imu, Magnetometer, AirPressure, NavSat, Sensors, etc.) are loaded at runtime via PX4's `server.config` (see `PX4-Autopilot/src/modules/simulation/gz_bridge/server.config`).

This requires sourcing PX4's `gz_env.sh` before launching Gazebo so that:
- `GZ_SIM_SERVER_CONFIG_PATH` points to `server.config` → plugins load
- `GZ_SIM_RESOURCE_PATH` includes PX4's models → model includes resolve (`x500`, `mono_cam`)

The `make gazebo-sentinel-world` target handles this automatically.

## Layout (top-down)

```
Y+
  |   [Office2]  [Warehouse]      [Tower]  [Garage]
  |       [Building H]   [Office]
  |
  |  Person4  Tree1    Person3  Tree2
  |  ======== SIDEWALK NORTH ========
  |  StopSign  Barrier   JerseyB  TrashCan
  |  =========== MAIN ROAD ===========  ← Y=0, lane markings
  |  TrafficLight   CarRed  CarBlack
  |  ======== SIDEWALK SOUTH ========
  |  Person6  Lamp2    Person2  Person5  Tree3
  |  Dumpster
  |       [Shop]   [Residential]    [Apartment]
  |
  |                           [CarSilver]
Y-  ----X-  ------X+  ----> X+
```

The drone spawns at the origin (0, 0, 0) and takes off vertically.

## Quick Start

### 1. Launch Gazebo (Terminal 1)

```bash
cd ons-sentinel-core
make gazebo-sentinel-world
```

Or manually (requires PX4 SITL built first):

```bash
source PX4-Autopilot/build/px4_sitl_default/rootfs/gz_env.sh
gz sim ons-sentinel-core/configs/gz/sentinel_street.sdf -r
```

### 2. Launch PX4 SITL (Terminal 2)

```bash
cd ons-sentinel-core
make px4-sentinel-world
```

Or manually:

```bash
cd PX4-Autopilot
PX4_GZ_WORLD=sentinel_street make px4_sitl gz_x500_mono_cam
```

### 3. View the Camera (Terminal 3)

```bash
cd ons-sentinel-core
make sentinel-camera-view
```

Or with full topic path:

```bash
python3 -m apps.tools.view_gazebo_camera \
  --topic /world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
```

**Controls:** `s` = save frame, `q` = quit

### Headless (no display)

```bash
make sentinel-camera-headless
# Saves latest frame to /tmp/gazebo_camera.jpg every second
```

### 4. Run Sentinel Pipeline

```bash
python3 -m apps.tools.run_realtime_loop \
  --config configs/sim_px4_gazebo_camera.yaml \
  --mission-id sentinel_street \
  --mode px4_sitl \
  --video-source-type gazebo_camera \
  --gazebo-camera-topic "/world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image" \
  --gazebo-world-name sentinel_street \
  --backend mock \
  --tak-mode dry_run \
  --max-frames 100 \
  --target-fps 10
```

## Camera Topic

The default topic pattern:

```
/world/<world_name>/model/<model_name_instance>/link/<link_name>/sensor/<sensor_name>/image
```

For sentinel_street (first instance):

```
/world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
```

Note the `_0` suffix — PX4 appends an instance number when spawning.

Discover topics:

```bash
make gazebo-topic-list
# or
gz topic --list | grep image
```

## Adding / Moving Objects

All objects are defined in `configs/gz/sentinel_street.sdf`.

### Add a new car

Copy an existing car block and change the `<pose>` and model name:

```xml
<model name="car_green">
  <static>true</static>
  <pose>X Y Z 0 0 YAW</pose>
  <link name="link">
    <collision name="collision"><geometry><box><size>4.2 1.8 1.3</size></box></geometry></collision>
    <visual name="body">
      <geometry><box><size>4.2 1.8 1.0</size></box></geometry>
      <material>
        <ambient>R G B 1</ambient>
        <diffuse>R G B 1</diffuse>
      </material>
    </visual>
    <visual name="cabin">
      <pose>0 0 0.65 0 0 0</pose>
      <geometry><box><size>2.0 1.6 0.7</size></box></geometry>
      <material><ambient>0.2 0.25 0.3 1</ambient><diffuse>0.25 0.3 0.35 1</diffuse></material>
    </visual>
  </link>
</model>
```

### Add a new person

```xml
<model name="person_N">
  <static>true</static>
  <pose>X Y 0 0 0 0</pose>
  <link name="link">
    <visual name="torso">
      <pose>0 0 0.9 0 0 0</pose>
      <geometry><cylinder><radius>0.18</radius><length>0.9</length></cylinder></geometry>
      <material><ambient>R G B 1</ambient><diffuse>R G B 1</diffuse></material>
    </visual>
    <visual name="head">
      <pose>0 0 1.55 0 0 0</pose>
      <geometry><sphere><radius>0.15</radius></sphere></geometry>
      <material><ambient>0.85 0.7 0.55 1</ambient><diffuse>0.9 0.75 0.6 1</diffuse></material>
    </visual>
  </link>
</model>
```

### Pose format

```
<pose>X Y Z  ROLL PITCH YAW</pose>
```

- X, Y: position in meters (origin = drone spawn)
- Z: height above ground (0 = ground level, set to half-height for boxes)
- YAW: rotation around Z axis in radians (e.g. 3.14 = 180 degrees, 1.57 = 90 degrees)

### Color format (RGB, 0.0 to 1.0)

```
<ambient>R G B 1</ambient>   — shadow color
<diffuse>R G B 1</diffuse>   — lit color (most visible)
<specular>R G B 1</specular> — highlight reflection
```

## Configuration

Override variables in the Makefile:

```bash
make gazebo-sentinel-world PX4_DIR=/path/to/PX4-Autopilot
make sentinel-camera-view SENTINEL_STREET_TOPIC=/world/my_world/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
```

## Troubleshooting

### "Accel/Gyro/Barometer/Compass missing"

PX4 can't find sensor data from Gazebo. Causes:

1. **Gazebo started without `gz_env.sh`** — system plugins (Imu, Magnetometer, etc.) not loaded. Fix: use `make gazebo-sentinel-world` which sources `gz_env.sh` automatically.

2. **`GZ_SIM_RESOURCE_PATH` missing PX4 models** — Gazebo can't resolve `<uri>x500</uri>` includes, so the model spawns without sensors. Fix: source `gz_env.sh` before starting Gazebo.

3. **PX4 can't find the world** — `PX4_GZ_WORLD=sentinel_street` looks in `PX4-Autopilot/Tools/simulation/gz/worlds/`. A symlink should exist there pointing to the actual SDF.

### Gazebo opens empty / grey world

Check that the SDF path is correct and the file exists:

```bash
ls -la ons-sentinel-core/configs/gz/sentinel_street.sdf
```

### Camera topic not found

1. Gazebo must be running with the Sensors plugin (loaded via server.config)
2. PX4 must have spawned the x500_mono_cam model
3. Run `make gazebo-topic-list` to discover available topics
4. The model name has an instance suffix: `x500_mono_cam_0`, not `x500_mono_cam`

### Low FPS / rendering issues

- Ensure GPU has enough VRAM: `nvidia-smi`
- Stop other GPU processes: `pkill -f llama-server`
- Use `--backend mock` for testing without YOLO inference

### Headless rendering

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```
