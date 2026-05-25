# Realistic ISR Simulation

Urban environment with moving pedestrians and vehicles for detection and tracking validation.

## Architecture

```
Gazebo (sentinel_street world)
├── 8 pedestrian actors (Fuel walk.dae mesh, scripted trajectories)
├── 2 standing people (Fuel Male visitor, FemaleVisitor)
├── 6 parked vehicles (Fuel models: Hatchback, Pickup, TruckBox, Bus, Prius)
├── 4 moving vehicles (Fuel models, moved by traffic script via set_pose)
├── 8 buildings, roads, sidewalks, obstacles, signs, lamps, trees
└── Camera sensor on x500_mono_cam drone (30 Hz, 1280x960)

Asset Downloader (apps/tools/download_gazebo_fuel_assets.py)
└── Downloads/verifies Fuel models to ~/.gz/fuel/ cache

Traffic Script (apps/tools/run_gazebo_traffic.py)
└── Moves vehicle_moving_1-4 via gz.transport13 /world/<name>/set_pose service

YOLO Test (apps/tools/run_gazebo_yolo_test.py)
└── Subscribes to camera topic → YOLO detection → prints stats

Recording (apps/tools/record_gazebo_camera.py)
└── Subscribes to camera topic → cv2.VideoWriter → MP4/JPEGs
```

## Launch Order

### Terminal 1: Gazebo + PX4
```bash
make gazebo-realistic-assets   # Download Fuel models (first time only)
make px4-sentinel-world-clean
```

Wait for PX4 console to show "pxh>"

### Terminal 2: Vehicle traffic
```bash
make gazebo-traffic
```

### Terminal 3: Camera viewer
```bash
make sentinel-camera-view
```

### Terminal 4 (optional): Record
```bash
make gazebo-record DURATION=30 OUTPUT=/tmp/demo.mp4
```

### Terminal 5 (optional): YOLO test
```bash
make gazebo-yolo-test BACKEND=yolo FRAMES=100
```

### Quick demo (just prints instructions)
```bash
make sentinel-actor-demo
```

## Scene Contents

| Category | Count | Type |
|----------|-------|------|
| Walking pedestrians | 8 | Fuel walk.dae mesh actors with scripted trajectories |
| Standing people | 2 | Fuel Male visitor, FemaleVisitor `<include>` models |
| Parked vehicles | 6 | Fuel models (Hatchback x2, Pickup, TruckBox, Bus, Prius) |
| Moving vehicles | 4 | Fuel models moved by traffic script (Pickup, Hatchback, Prius, TruckBox) |
| Buildings | 8 | Static boxes (office, warehouse, residential, etc.) |
| Roads | 2 | Main east-west + cross north-south |
| Lane markings | 11 | Dashed center line |
| Crosswalks | 2 | At intersection |
| Trees | 3 | Trunk + canopy |
| Street lamps | 2 | With emissive heads |
| Traffic light | 1 | Red/yellow/green |
| Stop sign | 1 | Red cylinder |
| Barriers | 3 | Traffic + jersey barrier |
| Other | 2 | Dumpster, trash can |

## Fuel Assets

All people and vehicle models use Gazebo Fuel assets. On first launch Gazebo downloads them automatically, but pre-caching is recommended:

```bash
make gazebo-realistic-assets
```

This runs `apps/tools/download_gazebo_fuel_assets.py` which downloads all required models to `~/.gz/fuel/`.

### Required Models

| Model | Owner | Used As |
|-------|-------|---------|
| Male visitor | OpenRobotics | Standing person |
| FemaleVisitor | OpenRobotics | Standing person |
| Walking person | OpenRobotics | Collision mesh for pedestrian actors |
| actor | Mingfei | Walk animation mesh for pedestrian actors |
| Hatchback | OpenRobotics | Parked + moving vehicles |
| Pickup | OpenRobotics | Parked + moving vehicles |
| TruckBox | OpenRobotics | Parked + moving vehicles |
| Bus | OpenRobotics | Parked vehicle |
| Prius Hybrid with sensors | OpenRobotics | Parked + moving vehicles |

### Cache location
```
~/.gz/fuel/fuel.gazebosim.org/<owner>/models/<model_name>/
```

Cache names are lowercase, e.g. "Male visitor" → `male visitor/`.

### Verify cache
```bash
python3 -m apps.tools.download_gazebo_fuel_assets --verify-only
```

## Pedestrian Routes

| Actor | Route | Loop time |
|-------|-------|-----------|
| pedestrian_1 | East along north sidewalk | 20s |
| pedestrian_2 | West along south sidewalk | 18s |
| pedestrian_3 | Cross road via crosswalk | 12s |
| pedestrian_4 | Rectangle around intersection | 30s |
| pedestrian_5 | East along far south sidewalk | 14s |
| pedestrian_6 | Triangle near intersection | 20s |
| pedestrian_7 | Short patrol east sidewalk | 10s |
| pedestrian_8 | Between warehouse and west sidewalk | 16s |

## Moving Vehicle Routes

| Model | Fuel Model | Route | Speed |
|-------|------------|-------|-------|
| vehicle_moving_1 | Pickup | East lane | 8 m/s (~30 km/h) |
| vehicle_moving_2 | Hatchback | West lane | 8 m/s |
| vehicle_moving_3 | Prius Hybrid with sensors | East lane (slow) | 4 m/s (~15 km/h) |
| vehicle_moving_4 | TruckBox | West lane | 6 m/s |

## Performance Tuning

If FPS drops below 20:

1. Reduce pedestrian count: edit `configs/gz/sentinel_street.sdf`, remove `pedestrian_5` through `pedestrian_8`
2. Reduce moving vehicles: remove `vehicle_moving_3` and `vehicle_moving_4` from SDF, and edit `run_gazebo_traffic.py` `_default_routes()`
3. Lower traffic script rate: `make gazebo-traffic` → add `--hz 10`
4. Reduce camera resolution: edit `PX4-Autopilot/Tools/simulation/gz/models/mono_cam/model.sdf`

Expected performance on a modern GPU:
- Camera FPS: 25-30 Hz
- Gazebo RTF: 0.7-1.0
- With 8 actors + 4 moving vehicles: ~20-25 FPS

## Diagnostics

```bash
make gazebo-camera-rate    # Check camera topic Hz
make gazebo-sim-stats      # Check sim time, RTF, model count
make gpu-status            # Check GPU utilization
```

## Troubleshooting

### Pedestrians not visible
- First launch downloads Fuel mesh (~5 MB) — requires network
- Pre-cache with `make gazebo-realistic-assets`
- Cached at `~/.gz/fuel/`
- Check Gazebo console for download errors
- Verify actor `<skin>` URL is reachable

### Fuel models appear as boxes / missing meshes
- Run `python3 -m apps.tools.download_gazebo_fuel_assets --verify-only`
- If models are missing, run `make gazebo-realistic-assets` to download
- Check network connectivity to `fuel.gazebosim.org`
- Fuel CLI required: `gz fuel download -u <url>`

### Vehicles not moving
- Traffic script requires Gazebo running first
- Check `make gazebo-sim-stats` shows `models: 17+`
- Script uses `set_pose` service — requires UserCommands plugin (loaded by PX4 server.config)

### Camera FPS drops
- Run `make gpu-status` — check VRAM usage
- Kill llama-server or other GPU processes
- Reduce actor count (see Performance Tuning)

### Duplicate Gazebo instances
```bash
make gazebo-kill-stale
make px4-sentinel-world-clean
```

## Adding New Vehicles/Pedestrians

### Add a parked vehicle
Add a Fuel `<include>` to `configs/gz/sentinel_street.sdf`:
```xml
<include>
  <name>my_car</name>
  <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Hatchback</uri>
  <pose>X Y Z 0 0 YAW</pose>
  <static>true</static>
</include>
```
Replace `Hatchback` with any Fuel model name. Pre-cache with `make gazebo-realistic-assets` or let Gazebo download on first launch.

### Add a walking pedestrian
Add an `<actor>` block to `configs/gz/sentinel_street.sdf`:
```xml
<actor name="my_pedestrian">
  <skin>
    <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
    <scale>1.0</scale>
  </skin>
  <animation name="walk">
    <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
    <interpolate_x>true</interpolate_x>
  </animation>
  <script>
    <loop>true</loop>
    <delay_start>0.0</delay_start>
    <auto_start>true</auto_start>
    <trajectory id="0" type="walk" tension="0.6">
      <waypoint><time>0</time><pose>X1 Y1 1.0 0 0 YAW1</pose></waypoint>
      <waypoint><time>T</time><pose>X2 Y2 1.0 0 0 YAW2</pose></waypoint>
      <!-- add more waypoints for loop -->
    </trajectory>
  </script>
</actor>
```

### Add a moving vehicle
1. Add a Fuel `<include>` to SDF at the initial route position:
```xml
<include>
  <name>vehicle_moving_5</name>
  <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Bus</uri>
  <pose>X Y Z 0 0 YAW</pose>
  <static>true</static>
</include>
```
2. Add a `Route` entry to `_default_routes()` in `apps/tools/run_gazebo_traffic.py` using the matching `model_name`
