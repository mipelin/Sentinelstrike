# Gazebo Camera Integration

## Data Flow

```
Gazebo sim (x500_mono_cam model)
  └─ gz-sim-sensors-system renders camera sensor
      └─ publishes gz.transport topic:
         /world/default/model/x500_mono_cam/link/camera_link/sensor/camera/image
          └─ GazeboCameraBridge (Python, gz.transport13 subscriber)
              └─ protobuf Image → numpy BGR array
                  └─ read_frame() → RealTimeMissionLoop
                      └─ YOLO → Tracking → Geolocalization → TAK
```

No GStreamer, no ROS2, no external processes. The bridge subscribes directly to Gazebo's transport layer via the `python3-gz-transport13` bindings.

## Design Rationale

**Why gz.transport13 over GStreamer UDP:**
- OpenCV in this environment has no GStreamer backend (`GStreamer: NO`)
- In-process Python subscription = lower latency than encoding/decoding through a network hop
- No extra pipeline processes to manage
- `python3-gz-transport13` and `python3-gz-msgs10` are already installed

**Why a separate GazeboCameraBridge class:**
- Keeps Gazebo dependency isolated from `ManagedVideoSource`
- Same duck-typed interface (`open/read_frame/close/health/write_health_json`)
- Factory returns the correct type based on `source_type`
- No changes to `ManagedVideoSource` or `loop.py`

**Why a custom world SDF:**
- PX4's `default.sdf` lacks the `gz::sim::systems::Sensors` plugin
- Without it, camera sensors are defined but never rendered
- `configs/gz/sentinel_camera.sdf` is `default.sdf` + Sensors plugin + standard PX4 plugins

**Why x500_mono_cam model:**
- Already exists in PX4-Autopilot (`Tools/simulation/gz/models/x500_mono_cam/`)
- Composes x500 airframe + mono_cam (1280x960 RGB, 30Hz, horizontal_fov 1.74 rad)
- Camera mounted at `(0.12, 0.03, 0.242)` relative to base_link, forward-facing
- No need to create a new model

## Launch Instructions

### Terminal 1: Start Gazebo with sensor rendering

```bash
cd PX4-Autopilot
PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=x500_mono_cam \
  gz sim ../ons-sentinel-core/configs/gz/sentinel_camera.sdf -r
```

### Terminal 2: Start PX4 SITL

```bash
cd PX4-Autopilot
PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=x500_mono_cam \
  make px4_sitl gz_x500_mono_cam
```

### Terminal 3: Run Sentinel

```bash
cd ons-sentinel-core
make realtime-gazebo-demo
```

Or manually:

```bash
python3 -m apps.tools.run_realtime_loop \
  --config configs/sim_px4_gazebo_camera.yaml \
  --mission-id realtime_gazebo \
  --mode px4_sitl \
  --video-source-type gazebo_camera \
  --gazebo-camera-topic "/world/default/model/x500_mono_cam/link/camera_link/sensor/camera/image" \
  --backend mock \
  --tak-mode dry_run \
  --max-frames 100 \
  --target-fps 10
```

### Quick demo (prints instructions only):

```bash
make gazebo-camera-demo
```

## Topic Discovery

If the camera topic name is different (e.g., different world name or model name):

```bash
make gazebo-topic-list
```

Or manually:

```bash
gz topic --list | grep image
```

The default topic pattern is:
```
/world/<world_name>/model/<model_name>/link/<link_name>/sensor/<sensor_name>/image
```

## Validation Metrics

The bridge reports the same health metrics as other video sources:

| Metric | Description |
|--------|-------------|
| `fps_estimate` | Actual frame rate from camera |
| `dropped_frames` | Frames produced by Gazebo but not consumed by loop |
| `connected` | Whether the gz transport subscription is active |
| `stale` | No frames received within `2/target_fps` seconds |
| `latest_frame_age_ms` | Time since last frame was read |

These are written to `runs/<run_dir>/sensor_health.json` after each run.

## Troubleshooting

### "gz.transport13 not available"

Install the bindings:
```bash
sudo apt install python3-gz-transport13 python3-gz-msgs10
```

### Camera topic shows no messages

1. Verify Gazebo is running with the Sensors plugin:
   ```bash
   gz topic --list | grep Sensors
   ```
   If empty, you're using the wrong world file. Use `configs/gz/sentinel_camera.sdf`.

2. Verify the model loaded with the camera:
   ```bash
   gz topic --list | grep camera
   ```
   Should show the image topic. If empty, `PX4_SIM_MODEL` may not be set to `x500_mono_cam`.

3. Check the exact topic name:
   ```bash
   make gazebo-topic-list
   ```

### Headless rendering

The Sensors plugin requires a rendering context. On headless servers:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

Or run with software rendering:
```bash
gz sim --render-engine-gui-api-backend opengl ...
```

### Low FPS / high latency

- The camera sensor runs at 30Hz by default (defined in `mono_cam/model.sdf`)
- The Sentinel loop target is set by `--target-fps`
- If YOLO inference is slow, FPS drops — use `--backend mock` for testing
- `dropped_frames` in health metrics means frames are arriving faster than consumed (expected)

## Jetson Migration Path

On Jetson hardware with a real camera, the Gazebo bridge is not needed. Replace with:

1. **RTSP source** (IP camera):
   ```yaml
   video:
     source_type: rtsp
     rtsp_url: rtsp://camera.local:554/stream
   ```

2. **MIPI CSI camera** (webcam source):
   ```yaml
   video:
     source_type: webcam
     webcam_index: 0
   ```

The `GazeboCameraBridge` is simulation-only. The rest of the pipeline (perception, tracking, geo, TAK) is unchanged regardless of video source.
