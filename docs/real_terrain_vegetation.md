# Real Terrain ISR Simulation

Lightweight vegetation, distinct people/vehicles, and ISR tracking on real-terrain
Gazebo worlds, designed for UAV search, tracking, and re-identification testing.

## Quick Start

```bash
cd ons-sentinel-core

# Generate all density variants (light, medium, heavy)
make realterrain-vegetation-all WORLD=1779343687303

# Launch with PX4
cd ../PX4-Autopilot
PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=x500_mono_cam \
  PX4_GZ_WORLD=1779343687303_veg_medium \
  make px4_sitl gz_x500_mono_cam

# Back in ons-sentinel-core: run ISR tracker test
make realterrain-isr-test

# Follow a specific target
make realterrain-isr-follow TGT=TGT-001 FMODE=yaw_xy

# Safe intercept mode (approach + orbit at standoff)
make realterrain-isr-intercept TGT=TGT-001

# SAFE standoff follow (observe at distance, no collision)
make realterrain-isr-standoff TGT=TGT-001 DIST=10 ALT=10 ORBIT=1
```

## Output Files

```
PX4-Autopilot/Tools/simulation/gz/worlds/
  1779343687303_veg_light.sdf
  1779343687303_veg_medium.sdf
  1779343687303_veg_heavy.sdf
```

The original terrain world (`1779343687303.sdf`) is never modified.

## Density Profiles

| Profile | Vegetation | People | Vehicles | Clusters | Shadows |
|---------|-----------|--------|----------|----------|---------|
| light   | ~30       | 4      | 3        | 2        | No      |
| medium  | ~100      | 8      | 5        | 4        | Yes     |
| heavy   | ~250      | 15     | 8        | 8        | Yes     |

## Distinct People

Each person has a **unique color combination** (shirt, pants, optional hat, backpack)
for re-identification testing. Person names are stable semantic IDs:

```
person_red_shirt, person_blue_jacket, person_green_vest, person_white_hat,
person_orange_pack, person_yellow_rain, person_gray_coat, person_purple_top,
person_khaki_shorts, person_black_hoodie, person_camo_pants, person_pink_scarf,
person_brown_jacket, person_teal_shirt, person_red_cap, person_maroon_sweat,
person_navy_suit, person_lime_tank, person_denim_jacket, person_beige_vest
```

Each person has a walking actor (Fuel mesh) plus a static colored inline marker
so YOLO sees unique colors from aerial view.

### Visual Distinctiveness

Each person varies along 6 dimensions:
- **Shirt color**: unique RGB per person
- **Pants color**: unique RGB per person (deduplicated)
- **Backpack**: small colored box visible from aerial overhead
- **Hat**: optional cylinder with unique color
- **Height**: 0.88-1.10 scale factor
- **Width**: 0.88-1.08 body width scale
- **Speed**: "slow" (5-8s/waypoint), "normal" (3-6s), "fast" (2-4s)

### Identity Descriptors

The ISR tracker extracts body-region color features and displays identity
descriptors in the overlay:

```
TGT-001 person red/dark_blue    ← upper_color/lower_color
TGT-002 person blue/gray
```

## Distinct Vehicles

Vehicles have unique types and colors:

```
sedan_red, sedan_blue, sedan_white,
pickup_black, pickup_green,
van_white, van_yellow,
suv_silver, suv_red,
truck_brown
```

## ISR Tracker

The ISR tracker (`--track isr`) provides:

- **Persistent IDs** through occlusions (max_lost_frames=90)
- **Camera motion compensation** for drone movement
- **Spatial-color appearance features** (149-dim) for re-identification
- **4+1-stage cascade matching** (BoT-SORT style + cross-lifecycle Re-ID)
- **Identity descriptors** (upper/lower body color names)
- **Semantic identity fingerprints** with multi-signal matching
- **Cross-lifecycle identity memory** (30s archive, reassigns archived IDs)
- **Predictive search** using last known position/heading
- **Track lifecycle**: TENTATIVE -> CONFIRMED -> LOST -> DELETED
- **World-space projection** — pixel-to-ground coordinate conversion via drone telemetry
- **Occlusion reasoning** — VISIBLE/OCCLUDED/SEARCHING/REACQUIRED/STALE states
- **Terrain-aware emergence prediction** — predicts where occluded targets will reappear
- **Geospatial trail storage** — persistent world-space target paths
- **Multi-hypothesis tracking** — probabilistic reasoning about lost targets (up to 5 hypotheses per target)
- **Group merge/split detection** — reduces catastrophic ID switches during pedestrian crossings
- **Hypothesis-weighted search** — probability-driven search sectors instead of simple sweeps

### Appearance Features

The 149-dim feature vector includes:
- 96-dim: 2x2 spatial grid HSV histograms
- 5-dim: global HSV statistics
- 24-dim: upper body color histogram (H + S channels)
- 12-dim: lower body color histogram
- 4-dim: shape features (aspect ratio, fill ratio, relative size)
- 8-dim: dominant color bins per body region

Gallery stores 15 snapshots with temporal weighting (recent features preferred)
and computes appearance stability score.

### Semantic Re-Identification

Each track maintains a `SemanticIdentity` fingerprint combining:
- **Appearance**: 149-dim features with gallery stability scoring
- **Spatial**: position, velocity, heading (extrapolated prediction)
- **Temporal**: hit count, track age, appearance consistency

Multi-signal similarity (default weights):
- Appearance: 50% (boosted to 65% for stable identities)
- Spatial: 30% (distance from predicted position)
- Heading: 20% (boosted for moving targets)

### Cross-Lifecycle Re-ID

When a track is DELETED, its identity is archived for 30 seconds.
New detections are checked against archived identities in Stage 5
of cascade matching. If a strong match is found (>0.6 similarity),
the original track ID is restored instead of assigning a new ID.

### Identity-Aware Re-Acquisition

When a tracked target disappears behind vegetation:

1. **SEARCHING** — scan toward predicted position (not just yaw rotation)
2. **CANDIDATE** — similar identity found but uncertain (sim 0.45-0.65)
3. **REACQUIRED** — high-confidence identity match (sim > 0.65)
4. **LOCKED** — confirmed re-acquisition

The tracker stores identity memory per target and prefers re-acquiring
the same visual identity over the nearest person.

### World-Space Tracking

When drone telemetry is available (via follow backend), the tracker projects
each target's bounding box to world-space coordinates:

- **World position**: lat/lon, north/east meters from drone, ground distance
- **World velocity**: speed in m/s, heading in degrees (from consecutive positions)
- **Uncertainty radius**: grows with distance, off-center angle, and shallow pitch
- **World trail**: up to 200 historical lat/lon positions per target

Projection uses tan(pitch) geometry with bbox bottom-center (feet position)
for accurate ground contact estimation.

### Occlusion Reasoning

Each track has an occlusion state machine:

```
VISIBLE → OCCLUDED → SEARCHING → STALE
                ↓           ↓
            REACQUIRED ←───┘
```

- **OCCLUDED**: target lost for 1-15 frames
- **SEARCHING**: target lost >15 frames, actively predicting emergence
- **REACQUIRED**: target re-detected in vegetation zone
- **STALE**: target lost >80% of max_lost_frames

Terrain-aware emergence prediction generates up to 5 hypotheses:
1. Velocity extrapolation (3 speed variants)
2. Nearest road/path exit points
3. Vegetation zone edge exits (weighted by heading direction)

### Geospatial Layer

The `GeospatialLayer` accumulates world-space target histories across frames:
- Target trails (up to 1000 points per target)
- Last-seen positions for all targets
- Grid-based activity heatmap (5m resolution)

### Multi-Hypothesis Tracking

When a target disappears, the tracker generates up to 5 probabilistic hypotheses
explaining what happened:

- **OCCLUDED** — hidden behind vegetation (boosted by terrain occlusion probability)
- **CONTINUED_PATH** — velocity extrapolation along roads/paths
- **EXITED_FOV** — left the camera frame (boosted near frame edges)
- **MERGED_GROUP** — merged with nearby same-class tracks (reduces ID switches)
- **STATIONARY** — target stopped moving (boosted by low speed history)

Each hypothesis has a probability weight that evolves over time:
- OCCLUDED decays slowly (vegetation can hide for seconds)
- EXITED_FOV grows if target doesn't emerge
- STATIONARY grows slowly (if target was stationary, it's probably still there)

The follow controller uses hypothesis-weighted search sectors instead of simple
directional sweeps. When MERGED_GROUP probability is high, it slows down and
waits for a clean split before resuming tracking.

Overlay displays hypothesis probability bars for each tracked target and
merge-group indicators connecting overlapping tracks.

## Follow Modes

### overlay (default)
Visual-only tracking with reticle and diagnostics. No offboard commands.

### yaw
Offboard yaw control to keep target horizontally centered.

### yaw_xy
Yaw + forward/back + lateral translation to:
- Keep target centered horizontally
- Maintain target at desired size/range
- Strafe laterally only when yaw is aligned
- Max speed conservative: 0.5-1.0 m/s

### intercept
Safe approach mode:
1. **APPROACHING** — move toward target at conservative speed (max 0.5 m/s)
2. **ORBITING** — at standoff distance (bbox height > 200px), orbit at 0.15 rad/s
3. Never intentionally collides with people/vehicles
4. Target visually marked as "intercepted"

### standoff
SAFE observation mode — approach slowly, maintain standoff distance, never collide:
1. **APPROACHING** — creep toward standoff at conservative speed (max `--max-vxy` m/s)
2. **SAFE** — at standoff distance, holding position
3. **ORBITING** — at standoff with `--orbit-on-arrival`, orbit at 0.15 rad/s
4. **RETREAT** — too close, backing away immediately
5. **HOLD** — target lost, all velocity zeroed, yaw-only search
6. Maintains standoff altitude above target
7. Optional deterrence marker circle with `--deterrence-marker`

```bash
make realterrain-isr-standoff TGT=TGT-001 DIST=10 ALT=10 ORBIT=1
```

## Overlay Information

The ISR overlay shows:
- Track ID + class + identity descriptor (upper/lower color)
- State: LOCKED / SEARCHING / REACQUIRED / CANDIDATE / LOST
- Trajectory tail (50 points, fading)
- Velocity vector
- Follow diagnostics panel:
  - Mode and offboard status
  - Yaw rate, error, target speed
  - VX/VY velocity commands
  - Identity descriptor + similarity score
  - PID state, confidence, scan direction
  - Lost timer, geofence status
  - Intercept state (APPROACHING/ORBITING/STANDOFF)
  - Standoff phase (APPROACHING/SAFE/ORBITING/RETREAT/HOLD)
  - Standoff distance target + altitude
  - Deterrence marker status

## Makefile Targets

```bash
# World generation
make realterrain-vegetation-assets    # Download/generate model assets
make realterrain-vegetation-light     # Generate light density world
make realterrain-vegetation-medium    # Generate medium density world
make realterrain-vegetation-heavy     # Generate heavy density world
make realterrain-vegetation-all       # Generate all three densities
make realterrain-perf-report          # Run performance report
make realterrain-terrain-audit        # Audit Z placement of entities

# ISR testing
make realterrain-isr-test             # Run ISR tracker on vegetation world
make realterrain-isr-follow           # Follow target TGT-XXX with yaw_xy
make realterrain-isr-intercept        # Safe intercept target TGT-XXX
make realterrain-isr-standoff         # SAFE standoff follow TGT-XXX
```

### Variables

| Variable                | Default                              | Description                      |
|-------------------------|--------------------------------------|----------------------------------|
| `WORLD`                 | `1779343687303`                      | Terrain world/model name         |
| `REALTERRAIN_SEED`      | `42`                                 | Random seed for reproducibility  |
| `REALTERRAIN_VEG_TOPIC` | `...1779343687303_veg_medium...`     | Gazebo camera topic              |
| `TGT`                   | `TGT-001`                            | Target ID for follow/intercept   |
| `FMODE`                 | `yaw_xy`                             | Follow mode (yaw/yaw_xy/standoff)|
| `DIST`                  | `10`                                 | Standoff distance (meters)       |
| `ALT`                   | `10`                                 | Standoff altitude (meters)       |
| `MAXVXY`                | `1.0`                                | Max horizontal velocity (m/s)    |
| `ORBIT`                 | (unset)                              | Set to `1` to orbit at standoff  |
| `DETER`                 | (unset)                              | Set to `1` for deterrence marker |
| `FRAMES`                | `200` (test) / `300` (follow)        | Max frames to process            |
| `CONF`                  | `0.3`                                | YOLO confidence threshold        |
| `DEVICE`                | `auto`                               | Inference device                 |

## Terrain Height Placement

All entities (vegetation, people, vehicles) are placed at correct ground-level
Z coordinates using the heightmap TIF. The tool includes a Z audit that warns
about floating or sunken entities.

## Performance Considerations

- **Light**: negligible RTF impact
- **Medium**: 2-5% RTF reduction
- **Heavy**: 10-20% RTF impact — for screenshots only

Vegetation and static people/vehicles are visual-only (no collision).
The main rendering cost is additional primitives per frame.

## Validation Checklist

- [ ] World loads in `gz sim` without errors
- [ ] PX4 spawns x500_mono_cam successfully
- [ ] Sensor topics publish (IMU, air_pressure, magnetometer, navsat, camera)
- [ ] RTF remains close to 1.0 (light/medium)
- [ ] People stand on terrain (no floating/sunken)
- [ ] People are visually distinguishable by color
- [ ] Vehicles are visually distinguishable
- [ ] ISR tracker assigns stable IDs
- [ ] Identity descriptor shows correct colors
- [ ] Lost target is reacquired with same ID via identity match
- [ ] Follow yaw_xy keeps target in frame
- [ ] Intercept mode stops at standoff and orbits
- [ ] Standoff mode approaches at safe speed, holds at distance
- [ ] Standoff mode zeros all velocity when target lost (HOLD)
- [ ] Standoff orbit works with `--orbit-on-arrival`
- [ ] Deterrence marker appears when at standoff with `--deterrence-marker`
- [ ] YOLO detects people near vegetation clusters
