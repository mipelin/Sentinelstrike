# ONS Sentinel Core

Modular autonomy, simulation, perception and mission orchestration layer for light UAVs in laboratory environments.

## What this phase implements

- Shared Pydantic data models (GeoPoint, Detection, Track, VehicleState, MissionRequest, etc.)
- YAML-based configuration system
- Central logging via loguru
- In-process event bus with wildcard subscriptions
- JSONL event recorder with per-run folders
- Mission Planner v1 — lawnmower and perimeter search patterns with metrics
- Autonomy supervisor (minimal state machine)
- MAVLink Bridge — mock backend + MAVSDK PX4 SITL backend
- Safety policy — ARM/TAKEOFF/real backend blocked by default
- Perception v1 — mock/YOLO backends, video source, detections JSONL, metrics
- **Tracker v1** — IoU-based multi-object tracking with track lifecycle
- **Geolocalizer v1** — flat-ground pinhole geolocalization from tracks + vehicle state
- **TAK Bridge v1** — CoT XML generation and UDP/dry-run transport for ATAK/TAK
- **Integrated Mission Pipeline v1** — end-to-end simulation: plan, MAVLink mock, perception, tracking, geoloc, TAK, report
- **Quality Gate v1** — artifact validation, JSONL checks, smoke tests, linting, type checking
- CLI demos (mission test, MAVLink mock, PX4 SITL, perception, tracking, geolocalization, TAK bridge, integrated pipeline)
- Full pytest suite

## What this phase does NOT implement

- ROS2 integration
- Jetson deployment or Docker
- Any weapon, kinetic, or offensive capability

## Installation

```bash
cd ons-sentinel-core
make install
```

Optional extras:

```bash
make install-mavlink      # PX4 SITL support
make install-perception   # YOLO support (ultralytics)
```

## Run demos

```bash
# Mission lifecycle demo (mock)
make demo

# MAVLink mock bridge demo
make mavlink-mock-demo

# Perception mock demo
make perception-demo

# Perception + Tracking demo
make tracking-demo

# Perception + Tracking + Geolocalization demo
make geolocalization-demo

# Full pipeline + TAK bridge demo (dry_run)
make tak-demo

# Integrated mission pipeline (end-to-end)
make pipeline-demo

# Edge Agent Runtime (mock mode — full orchestrator)
make edge-agent-demo

# Edge Agent Runtime (PX4 SITL mode)
make edge-agent-px4-demo
```

Each creates a run folder under `runs/` with `metadata.json`, `events.jsonl`, and module-specific outputs.

## PX4 SITL Integration

### Prerequisites

1. Install MAVSDK extra: `make install-mavlink`
2. Install PX4 Autopilot locally (see https://docs.px4.io/)

### Launch PX4 SITL

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

PX4 SITL listens for MAVSDK connections on UDP port 14540 by default. The config uses ``udpin://0.0.0.0:14540`` (the modern MAVSDK connection URL — ``udp://`` is deprecated).

### Run SITL test

```bash
make px4-sitl-demo
```

**Important**: PX4 SITL must already be running before executing ``make px4-sitl-demo``.

By default the SITL test generates a small local mission (80m x 80m, 30m altitude) around the PX4 SITL home position. This avoids the "First waypoint far away from home" warning that occurs when using the Portugal-based demo mission JSON.

| Flag | Default | Description |
|------|---------|-------------|
| `--local-sitl-mission` | enabled | Generate mission around SITL home |
| `--no-local-sitl-mission` | — | Use the mission JSON file instead |

```bash
# Use local SITL mission (default)
make px4-sitl-demo

# Use mission JSON manually
python -m apps.tools.run_px4_sitl_test --no-local-sitl-mission
```

SITL local mission parameters are configured in ``configs/sim_px4.yaml`` under the ``sitl_local_mission`` section.

## Run tests

```bash
make test
```

Tests do not require PX4, ultralytics, or external video files.

## Perception v1

The perception module processes video sources and produces normalized detections.

| Backend | Description |
|---------|-------------|
| `mock` | Deterministic dummy detections every 10 frames. No dependencies. |
| `yolo` | YOLO via ultralytics. Requires `make install-perception`. |

### Outputs

- `detections.jsonl` — one Detection per line
- `perception_metrics.json` — frame count, detection count, class distribution
- `annotated.mp4` — optional video with drawn bounding boxes

## Tracker v1

The tracker takes frame-by-frame detections and produces persistent tracks with stable IDs.

### Algorithm

`simple_iou` — greedy IoU-based association. For each frame:

1. Filter detections by confidence threshold.
2. Match detections to existing tracks by highest IoU (same class, above threshold).
3. Unmatched detections create new tracks (`trk_000001`, `trk_000002`, ...).
4. Unmatched tracks increment `lost_frames`. After `max_lost_frames` they become `terminated`.
5. Lost tracks that get re-matched become `active` again.

### Track lifecycle

```
Detection (no match) -> active -> lost (after missed frames) -> terminated (after max_lost_frames)
                                    ^                            |
                                    |__ reacquired (if re-matched)
```

### Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `iou_threshold` | 0.3 | Minimum IoU for track association |
| `max_lost_frames` | 15 | Frames before a lost track is terminated |
| `min_confidence` | 0.0 | Minimum detection confidence to track |

### Outputs

- `tracks.jsonl` — one Track per line per frame, with track_id, status, age, bbox
- `tracker_metrics.json` — total/active/lost/terminated counts, class distribution

### Events

| Event | When |
|-------|------|
| `track_created` | New track created |
| `tracker_updated` | Every frame after update |
| `track_reacquired` | Lost track matches again |
| `track_terminated` | Track exceeds max_lost_frames |
| `tracking_completed` | Runner closes |

### Demo

```bash
make tracking-demo
```

### Limitations

- No Kalman filter or motion prediction.
- No ReID embeddings for appearance matching.
- No geolocation of tracks.
- No ByteTrack or DeepSORT yet.

## Geolocalizer v1

The geolocalizer converts tracked detections into estimated ground positions using a flat-ground pinhole camera model.

**Inputs**: Track (bbox), VehicleState (position, altitude, heading), CameraModel (FOV, pitch).

**Method**: `flat_ground_pinhole_v1` — estimates ground distance from altitude and camera pitch, projects pixel offset to bearing, and offsets the vehicle position. All estimates are approximate.

**Outputs**:

- `geo_observations.jsonl` — one GeoObservation per line
- `geolocalizer_metrics.json` — observation count, class distribution, average confidence/accuracy

### Demo

```bash
make geolocalization-demo
```

### Limitations

- Assumes flat terrain at `assumed_ground_alt_m`.
- No terrain elevation model (DEM).
- No real camera calibration or distortion correction.
- No gimbal telemetry integration.
- Accuracy is approximate and should not be used for precision claims.
- No multi-view triangulation or sensor fusion.

## TAK/ATAK Bridge v1

The TAK bridge converts system state into Cursor-on-Target (CoT) XML messages for ATAK/FiTAK/TAK servers.

**What it sends**:

| Input | CoT type | Description |
|-------|----------|-------------|
| VehicleState | `a-f-A-M-F-Q` | UAV position marker |
| GeoObservation | `a-u-G` | Observed object on ground |
| MissionPlan waypoints | `b-m-p-w` | Mission waypoint markers |
| SystemEvent | `b-a-o-tbl` | System alert |

**Modes**:

| Mode | Description |
|------|-------------|
| `dry_run` | Stores messages in memory. No network. Safe for tests. |
| `udp` | Sends CoT XML as UTF-8 UDP datagrams to configured host:port. |

### Demo

```bash
make tak-demo
```

### Connecting to ATAK/FreeTAKServer

Set `tak.mode: udp` and configure `cot_host`/`cot_port` to your TAK server's CoT input endpoint.

### Outputs

- `tak_messages.jsonl` — one record per message with XML payload
- `tak_metrics.json` — message count by category and type

### Limitations

- No TLS or authentication.
- No ATAK plugin (server-side CoT only).
- No bidirectional commands from ATAK to drone.
- No chat, mission packages, or protobuf.
- CoT types are approximate (not doctrinally precise).

## Integrated Mission Pipeline v1

A single command that runs the full end-to-end simulation pipeline:

1. **Plan** — MissionPlanner generates waypoints from mission JSON.
2. **MAVLink** — Mock bridge executes connect/upload/start/hold/return/land.
3. **Perception** — Processes video frames with mock/YOLO backend.
4. **Tracking** — IoU tracker produces persistent tracks.
5. **Geolocalization** — Estimates ground positions from tracks + vehicle state.
6. **TAK Bridge** — Publishes CoT markers (dry_run by default).
7. **Report** — Generates `pipeline_summary.json` and `report.md`.

### Demo

```bash
make pipeline-demo
```

### Outputs

All artifacts in a single `runs/<timestamp>_<mission_id>/` folder:

- `mission_plan.json`, `detections.jsonl`, `tracks.jsonl`
- `geo_observations.jsonl`, `tak_messages.jsonl`
- `pipeline_summary.json`, `report.md`
- `events.jsonl`, `metadata.json`
- Module-specific metrics JSON files

### Defaults

Uses mock MAVLink, mock perception, and dry_run TAK by default. No hardware, PX4, ultralytics, or ATAK required.

## Quality Gate v1

Validates that pipeline runs produce correct, parseable artifacts.

### Commands

```bash
make install-dev           # install with lint/typecheck tools
make lint                  # ruff lint
make format                # ruff format
make typecheck             # mypy (soft mode)
make quality               # validate latest run artifacts (strict)
make smoke                 # validate without strict exit
make pipeline-quality-demo # run pipeline + quality gate
make all-checks            # test + lint + typecheck
```

### What it checks

- **Smoke**: project files exist, core imports work.
- **Artifacts**: all expected files present in run dir.
- **JSON/JSONL**: every file is parseable.
- **Report**: `report.md` exists and is non-empty.
- **Summary**: `pipeline_summary.json` has required structure.

Strict mode (default) exits with code 1 on any failure. Non-strict prints warnings only.

## Architecture

```
sentinel/
  common/               # Shared types, events, event bus, logging, geo utils
  config/               # YAML loader + Pydantic config schema
  recorder/             # JSONL event recorder
  mission_planner/      # Patterns, validation, metrics, planner orchestrator
  autonomy_supervisor/  # State machine for mission lifecycle
  mavlink_bridge/       # Backend protocol, mock, MAVSDK, safety, bridge orchestrator
  perception/           # Video source, mock/YOLO backends, drawing, metrics, runner
  tracker/              # IoU tracker, track state, metrics, runner
  geolocalizer/         # Flat-ground pinhole geolocalizer, camera model, metrics, runner
  tak_bridge/            # CoT XML builder, mapper, transport (dry_run/UDP), bridge orchestrator
  pipeline/              # Integrated mission pipeline, summary, report
  edge_agent/            # Master runtime orchestrator: lifecycle, context, health
  quality/               # Artifact validation, JSONL checks, smoke tests
  dashboard/             # Tactical replay dashboard: FastAPI, Leaflet map, timeline, loader
  evidence/              # Mission evidence packaging: checksums, manifest, summaries, zip export
  sensors/               # Real sensor ingestion: file, webcam, RTSP with health + reconnect
apps/
  tools/                # CLI tools
  edge_agent/           # Edge Agent Runtime CLI
configs/                # YAML config files
missions/               # Mission definition JSON files
data/videos/            # Sample video files
```

## Edge Agent Runtime v1

The Edge Agent Runtime is the master operational orchestrator — a single process that runs a complete mission end-to-end: planning, MAVLink, perception, tracking, geolocalization, TAK/C2, reporting and health.

### Difference with `pipeline-demo`

`make pipeline-demo` runs the `IntegratedMissionPipeline` which is a linear orchestrator focused on the data pipeline. The Edge Agent Runtime adds:

- **Lifecycle state machine** with auditable transitions and events.
- **Context management** for all subsystem references.
- **Health snapshots** for monitoring.
- **PX4 SITL mode** with local mission generation and safe-land on failure.
- **Separation of concerns** — each subsystem is managed through `EdgeAgentContext`.

### Modes

| Mode | Description |
|------|-------------|
| `mock` | Uses MockMavlinkBackend. No hardware or PX4 required. Default. |
| `px4_sitl` | Uses MAVSDK backend against PX4 SITL. Requires PX4 running. |

### Commands

```bash
# Mock mode (no dependencies)
make edge-agent-demo

# PX4 SITL mode (requires PX4 running + make install-mavlink)
make edge-agent-px4-demo
```

### CLI

```bash
python -m apps.edge_agent.run_edge_agent \
  --config configs/sim.yaml \
  --mission missions/demo_search_area.json \
  --mode mock \
  --backend mock \
  --tak-mode dry_run \
  --max-frames 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `mock` | `mock` or `px4_sitl` |
| `--backend` | `mock` | Perception backend: `mock` or `yolo` |
| `--tak-mode` | `dry_run` | TAK transport: `dry_run` or `udp` |
| `--max-frames` | `30` | Max perception frames |
| `--local-sitl-mission` | enabled | Generate local mission around SITL home |
| `--vehicle-lat/lon/alt/heading` | 38/-8/80/90 | Simulated vehicle position |

### Outputs

All artifacts in `runs/<timestamp>_<mission_id>/`:

- `mission_plan.json` — planned waypoints
- `detections.jsonl` — perception detections
- `tracks.jsonl` — tracking output
- `geo_observations.jsonl` — geolocalized positions
- `tak_messages.jsonl` — C2 CoT messages
- `pipeline_summary.json` — mission summary
- `report.md` — markdown report
- `edge_agent_health.json` — runtime health snapshot
- `events.jsonl` — auditable event log

### Lifecycle events

The runtime publishes events for each state transition:

`edge_agent_started` → `edge_agent_state_changed` (per transition) → `edge_agent_completed` (or `edge_agent_failed`)

### Safety defaults

- Uses MAVLink mock by default — no hardware control.
- TAK is `dry_run` by default — no network traffic.
- PX4 SITL mode requires explicit `--mode px4_sitl`.
- On failure, attempts safe return-home and land if PX4 SITL is active.
- No ATAK-to-drone command processing.
- No control of physical UAV hardware.

## Modes

- **SIMULATION_MODE** — no hardware or PX4 SITL
- **EDGE_MODE** — on-device inference (future)
- **INTEGRATION_MODE** — connected to real autopilot (future)

## Demo Scenario Pack v1

Reproducible mission scenarios for testing, demonstration, and evidence generation. Each scenario runs the full pipeline (perception, tracking, geolocalization, operator gate, safety executor, TAK) with specific conditions.

### Commands

| Command | Scenario | Description |
|---------|----------|-------------|
| `make demo-list` | — | List all available scenarios |
| `make demo-observation` | `observation_confirmed` | Full pipeline with operator confirmation |
| `make demo-abort` | `operator_abort` | Operator abort triggers HOLD + LAND |
| `make demo-low-battery` | `low_battery_return` | Low battery triggers RETURN_HOME |
| `make demo-link-loss` | `link_loss_return` | Link loss triggers RETURN_HOME |
| `make demo-px4-live` | `px4_sitl_live_observation` | PX4 SITL live (requires PX4) |
| `make demo-all-safe` | All non-PX4 | Run all safe scenarios sequentially |

### CLI

```bash
python -m apps.tools.run_demo_scenario list
python -m apps.tools.run_demo_scenario run observation_confirmed
python -m apps.tools.run_demo_scenario run operator_abort
```

### Per-scenario outputs

Each scenario creates a `runs/<timestamp>_demo_<scenario_id>/` folder containing:

- `scenario_metadata.json` — scenario ID, name, execution timestamp, metrics
- `events.jsonl` — full event audit trail
- `realtime_metrics.json` — frame counts, FPS, timing
- `detections.jsonl`, `tracks.jsonl`, `geo_observations.jsonl`
- `operator_decisions.jsonl` — when operator gate is active
- `safety_actions.jsonl` — when safety triggers fire
- `tak_messages.jsonl` — CoT messages

## Tactical Replay Dashboard v1

A local web dashboard for post-mission replay, audit, and situational awareness. Read-only — no commands are sent to the vehicle.

### Quick start

```bash
# 1. Generate a run
make realtime-demo

# 2. Launch dashboard
make dashboard-local

# 3. Open browser
# http://127.0.0.1:8787
```

### Commands

| Command | Description |
|---------|-------------|
| `make dashboard` | Launch on 0.0.0.0:8787 |
| `make dashboard-local` | Launch on 127.0.0.1:8787 |

### Features

- **Run selector** — pick any completed run from `runs/`
- **Map** — Leaflet map with waypoints, UAV position, geo-observations (green = confirmed, orange = unconfirmed), safety events
- **Timeline** — unified chronological view of system events, operator decisions, safety actions, and geo-observations
- **Metrics panel** — detection/track/observation/TAK/operator/safety counts + mission summary
- **Report viewer** — rendered mission report markdown
- **Live mode** — auto-refreshes latest run data via WebSocket

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/runs` | List all runs |
| `GET /api/runs/latest` | Latest run summary |
| `GET /api/runs/{run_id}` | Run artifacts + summary |
| `GET /api/runs/{run_id}/timeline` | Unified timeline |
| `GET /api/runs/{run_id}/map` | Map layers (waypoints, observations, UAV) |
| `GET /api/runs/{run_id}/report` | Mission report (markdown) |
| `GET /api/health` | Health check |
| `WS /ws/runs/{run_id}/live` | Live updates for a specific run (0.5s) |
| `WS /ws/live/latest` | Live updates for latest run (0.5s) |

## Live Dashboard WebSocket v1

The dashboard supports live mode — watching a mission in progress in near real-time. Uses byte-offset JSONL tailing for efficient incremental updates.

### How to use

```bash
# Terminal 1: start dashboard
make dashboard-local

# Terminal 2: start a mission
make realtime-demo

# Browser: http://127.0.0.1:8787
# Click "Live Latest" button
# Timeline, map, and metrics update every 0.5 seconds
```

### Commands

| Command | Description |
|---------|-------------|
| `make dashboard-demo` | Print instructions for live demo |
| `make dashboard-local` | Start dashboard on 127.0.0.1:8787 |
| `make dashboard` | Start dashboard on 0.0.0.0:8787 |

### How it works

1. `JsonlTailer` tracks byte offset per JSONL file — only reads new lines since last poll
2. `LiveRunSession` tails all artifact JSONLs (events, detections, tracks, geo_observations, operator_decisions, safety_actions, tak_messages)
3. WebSocket endpoints poll every 0.5s and push updates: timeline tail, map layers, artifact counts
4. Frontend shows LIVE badge with pulse animation, auto-scrolls timeline, updates map incrementally

### Limitations

- Read-only — no drone control from the UI
- No authentication or RBAC
- No database — reads from filesystem JSON/JSONL
- No real-time video stream
- 0.5s polling interval (not event-driven)
- No replay speed control yet

## Mission Evidence Package v1

Every run automatically generates a `mission_package/` directory with a portable, checksummed evidence bundle.

### What it produces

Inside each `runs/<run_id>/mission_package/`:

| File | Description |
|------|-------------|
| `summary.json` | Top-level run summary: metadata, scenario, artifact line counts |
| `metrics_summary.json` | Aggregated per-module metrics (perception, tracker, geolocalizer, TAK, safety) |
| `operator_summary.json` | Operator decision counts by action and operator ID |
| `safety_summary.json` | Safety action counts by trigger and action, success flag |
| `timeline.json` | Consolidated chronological timeline from all event JSONLs |
| `replay_manifest.json` | Artifact inventory with SHA-256 hashes and sizes |
| `artifacts_index.json` | Artifacts grouped by category (events, perception, tracking, etc.) |
| `checksums.json` | SHA-256 checksums of all package files |

### Zip export

```python
from sentinel.evidence.exporter import export_zip
export_zip(Path("runs/<run_id>"))  # creates mission_package.zip
```

### Demo

```bash
make evidence-demo
```

### Integration

The demo scenario runner calls `build_evidence_package()` automatically after each scenario. The edge agent runtime and integrated pipeline can also call it manually.

## Real Sensor Ingestion v1

Unified video source supporting local files, USB webcams, and RTSP streams with automatic reconnect and health monitoring.

### Source types

| Type | Description |
|------|-------------|
| `file` | Local video file (default). Same behavior as before. |
| `webcam` | USB webcam via OpenCV `VideoCapture(index)`. No crash if camera missing. |
| `rtsp` | RTSP stream with automatic reconnect on disconnect. |

### Configuration

```yaml
video:
  source_type: file          # file | webcam | rtsp
  file_path: data/videos/demo.mp4
  webcam_index: 0
  rtsp_url: ""
  reconnect_enabled: true
  reconnect_interval_s: 5.0
  frame_width: null          # optional resolution override
  frame_height: null
  target_fps: 5.0
```

### Commands

```bash
# File source (default, unchanged)
make realtime-demo

# USB webcam
make realtime-webcam-demo

# RTSP stream
make realtime-rtsp-demo
```

### CLI

```bash
python -m apps.tools.run_realtime_loop \
  --video-source-type webcam \
  --webcam-index 0 \
  --backend yolo \
  --max-frames 100

python -m apps.tools.run_realtime_loop \
  --video-source-type rtsp \
  --rtsp-url "rtsp://192.168.1.100:554/stream" \
  --backend yolo \
  --max-frames 200
```

### Health monitoring

When using webcam or RTSP sources, the loop writes `sensor_health.json` to the run directory:

```json
{
  "connected": true,
  "fps_estimate": 14.5,
  "dropped_frames": 0,
  "reconnect_count": 0,
  "last_frame_utc": "2026-05-11T12:00:00Z",
  "stale": false,
  "source_type": "webcam",
  "source_label": "webcam:0"
}
```

### Events

| Event | When |
|-------|------|
| `sensor_connected` | Source opened successfully |
| `sensor_disconnected` | Read failure detected (RTSP/webcam) |
| `sensor_reconnected` | Reconnection succeeded |

### Troubleshooting

- **Webcam not found**: Check `ls /dev/video*`. Ensure user is in the `video` group. Try different `--webcam-index` values.
- **RTSP timeout**: Set `reconnect_interval_s` higher. Verify URL with `ffplay rtsp://...`.
- **Low FPS**: Reduce `frame_width`/`frame_height`. Use `--backend mock` to test without YOLO overhead.

### Limitations

- No CSI camera support (Jetson-specific).
- No GStreamer pipelines.
- No CUDA/TensorRT optimization.
- No audio.
- RTSP reconnect blocks the loop during retry.

## Dashboard UAV Telemetry Layer v1

Live UAV position, track, and telemetry visualization on the dashboard map.

### What it shows

- **UAV icon** — red triangle rotated by heading, moving on the map
- **Track polyline** — yellow line showing the UAV's flight path
- **Telemetry panel** — altitude, speed, heading, battery, mode, armed/disarmed, stale flag
- **Popup** — click the UAV icon for full telemetry details
- **Live follow** — in live mode, the map pans to follow the UAV

### Artifacts

The realtime loop writes `vehicle_states.jsonl` with one entry per frame:

```json
{
  "timestamp_utc": "2026-05-11T12:00:00Z",
  "vehicle_id": "uav_001",
  "position": {"lat": 38.001, "lon": -8.001, "alt_m": 80.0},
  "heading_deg": 90.0,
  "groundspeed_mps": 10.0,
  "battery_pct": 94.0,
  "mode": "AUTO",
  "armed": true,
  "stale": false
}
```

### Mock mode

In mock mode (no PX4), the UAV follows a simulated path:
- Uses `MockMovementTelemetryProvider` with waypoint interpolation
- Heading changes coherently toward next waypoint
- Battery decreases slowly
- Position updates every frame

### PX4 mode

With PX4 SITL (`make realtime-px4-demo`), uses real MAVLink telemetry from `MavlinkTelemetryProvider`. Stale telemetry is flagged.

### Commands

```bash
make realtime-demo        # mock movement
make dashboard-local      # view UAV on map
```

### Dashboard features

| Feature | Description |
|---------|-------------|
| UAV icon | Red triangle, rotated by heading |
| Track line | Yellow polyline showing flight path |
| Telemetry panel | Alt, speed, heading, battery, mode, armed, last update |
| Live follow | Map auto-pans to follow UAV |
| Stale indicator | Yellow "STALE" when telemetry is outdated |

### Limitations

- Read-only — no drone control from dashboard
- Mock movement is simplified (no wind, no physics)
- No 3D view or altitude visualization on map

## Upcoming modules

- **TLS TAK** — encrypted CoT transport
- **ATAK Plugin** — bidirectional Android integration
