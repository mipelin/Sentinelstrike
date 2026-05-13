# Architecture

## Overview

ONS Sentinel Core is built as a modular Python package where each subsystem owns its domain and communicates through a central event bus. This keeps modules decoupled and testable in isolation.

## Data flow

```
Mission JSON  ──>  MissionPlanner  ──>  MissionPlan
                                            │
AutonomySupervisor  <────────────────────────┘
       │
       ├─ publishes SystemEvent ──>  EventBus  ──>  MissionRecorder (JSONL)
       └─ state transitions drive mission lifecycle

Video Source  ──>  PerceptionRunner  ──>  Backend (mock / yolo)
                          │
                          ├─ detections ──>  TrackingRunner  ──>  SimpleIoUTracker
                          │                                          │
                          │                    ┌─────────────────────┘
                          │                    ├─ tracks.jsonl
                          │                    ├─ tracker_metrics.json
                          │                    └─ events (track_created, tracker_updated, etc.)
                          │
                          ├─ publishes SystemEvent ──>  EventBus  ──>  MissionRecorder (JSONL)
                          ├─ writes detections.jsonl
                          ├─ writes perception_metrics.json
                          └─ optionally writes annotated.mp4

MavlinkBridge  ──>  SafetyPolicy  ──>  Backend (mock / mavsdk)
       │
       └─ publishes SystemEvent ──>  EventBus  ──>  MissionRecorder (JSONL)
```

## Tracker

The tracker takes frame-by-frame `Detection` objects and produces persistent `Track` objects with stable IDs.

### Components

- **SimpleIoUTracker** — greedy IoU-based association engine. No external dependencies.
- **TrackState** — internal mutable track representation (age, lost frames, status).
- **TrackingRunner** — wraps tracker with event publishing, JSONL output, and metrics.
- **IoU** — bounding box intersection-over-union computation.
- **metrics** — track counts by status, class distribution, average track age.

### Association algorithm

1. Filter detections by `min_confidence`.
2. Build greedy matches: for each detection, find the best unmatched track with same `class_name` and IoU above `iou_threshold`.
3. Matched tracks update bbox, confidence, reset `lost_frames`.
4. Unmatched detections create new tracks with sequential IDs (`trk_000001`).
5. Unmatched tracks increment `lost_frames`. Beyond `max_lost_frames` they become `terminated`.
6. Lost tracks that re-match become `active` again.

### Track lifecycle

```
new detection ──> active ──> lost (missed frames) ──> terminated (exceeded max_lost_frames)
                               ^                         |
                               └── reacquired (re-matched)
```

### Events

| Event | When |
|-------|------|
| `track_created` | New track from unmatched detection |
| `tracker_updated` | Every frame after update |
| `track_reacquired` | Lost track matches a detection |
| `track_terminated` | Track exceeds max_lost_frames |
| `tracking_completed` | Runner closes |

### Integration with Perception

`TrackingRunner` is injected into `PerceptionRunner` as an optional `tracker_runner` parameter. During each frame, after detections are produced, the runner calls `tracker_runner.process_frame_detections()`. This keeps the tracker decoupled — it can also run standalone on a `detections.jsonl` file.

### Future work

- **Kalman filter** — motion prediction for smoother association under occlusion.
- **DeepSORT / ByteTrack** — appearance-based ReID embeddings for robust ID matching.
- **Geolocalizer** — project track positions to world coordinates using gimbal telemetry.
- **TAK markers** — push geolocated tracks as CoT markers to ATAK.

## Perception

The perception module processes video frames and produces normalized `Detection` objects.

### Components

- **VideoSource** — reads frames from file or webcam via OpenCV.
- **PerceptionBackend** (protocol) — `detect_frame()` returns a list of `Detection`.
- **MockPerceptionBackend** — deterministic dummy detections every 10 frames.
- **YoloPerceptionBackend** — wraps ultralytics YOLO. Lazy-imported.
- **PerceptionRunner** — orchestrates video source, backend, tracking, events, and output files.

## Mission Planner

### Processing flow

1. Validate area — minimum 3 points, non-degenerate bounding box.
2. Generate pattern — lawnmower (bounding-box rows) or perimeter (boundary loop).
3. Build route — launch point → pattern points → return to launch.
4. Compute metrics — total distance via haversine, estimated duration.
5. Validate plan — check waypoint count, distance, and altitude against constraints.

## MAVLink Bridge

The bridge decouples mission logic from autopilot hardware:

- **MavlinkBridge** receives commands and delegates to a backend.
- **SafetyPolicy** validates every command before execution.
- **MockMavlinkBackend** simulates state changes in memory.
- **MavsdkBackend** connects to PX4 SITL via MAVSDK-Python.

## Operating modes

| Mode | Description |
|------|-------------|
| `SIMULATION_MODE` | Mock backend or PX4 SITL. No real hardware. Used for development and CI. |
| `EDGE_MODE` | Runs on Jetson or companion computer. Real inference, simulated or real flight. |
| `INTEGRATION_MODE` | Connected to a real autopilot via MAVLink. Full hardware-in-the-loop. |

## Design principles

- **Modularity** — Each subsystem is a Python package with clear boundaries.
- **Human-in-the-loop** — Critical transitions require explicit operator confirmation.
- **Evidence-first** — Every state transition publishes a `SystemEvent`. The recorder guarantees a complete audit trail.
- **Edge-first** — Targets constrained hardware (Jetson Nano/Xavier). Pure Python where possible.
- **Safety-first** — Dangerous operations are off by default and must be explicitly enabled in configuration.

## Geolocalizer

The geolocalizer converts tracked detections into estimated ground positions.

### Data flow

```
Track (bbox) + VehicleState (position, alt, heading) + CameraModel (FOV, pitch)
    │
    FlatGroundGeolocalizer
    │
    ├─ pixel center of bbox → horizontal/vertical angle offsets
    ├─ camera pitch + vertical offset → ground distance via trigonometry
    ├─ vehicle heading + yaw offset + horizontal angle → bearing
    ├─ ground distance + bearing → north/east offset
    └─ offset_point() → GeoObservation with estimated_location

GeoObservation ──> geo_observations.jsonl
                ──> geolocalizer_metrics.json
                ──> EventBus (geo_observation_created, geolocalizer_updated, geolocalization_completed)
```

### Components

- **CameraModel** — Pydantic model: resolution, FOV, pitch/yaw/roll, mounting type.
- **FlatGroundGeolocalizer** — core algorithm: pinhole projection onto flat terrain.
- **GeolocalizationRunner** — wraps geolocalizer with event publishing, JSONL output, metrics.

### Assumptions

- Flat terrain at configured `assumed_ground_alt_m`.
- Pinhole camera with linear FOV mapping (no distortion).
- Camera pitch is fixed relative to vehicle body.
- Vehicle heading is true-north referenced.

### Events

| Event | When |
|-------|------|
| `geo_observation_created` | Each track successfully geolocated |
| `geolocalizer_updated` | After each batch of tracks processed |
| `geolocalization_completed` | Runner closes |

### Future work

- **Gimbal telemetry** — real-time camera orientation from MAVLink gimbal messages.
- **DEM / terrain elevation** — replace flat-ground assumption.
- **Multi-view triangulation** — combine observations from different angles.
- **Sensor fusion** — integrate with IMU or other positioning sources.
- **TAK bridge** — push GeoObservations as CoT markers to ATAK.

## TAK Bridge

The TAK bridge converts domain objects into Cursor-on-Target (CoT) XML messages and delivers them to ATAK/TAK endpoints.

### Data flow

```
VehicleState  ──> vehicle_state_to_cot()    ──> CoT XML (a-f-A-M-F-Q)
GeoObservation ──> geo_observation_to_cot() ──> CoT XML (a-u-G)
MissionPlan   ──> waypoints_to_cot()        ──> CoT XML (b-m-p-w) per waypoint
SystemEvent   ──> system_event_to_cot()     ──> CoT XML (b-a-o-tbl)

CoT XML ──> TakBridge ──> Transport (DryRun / UDP)
                      ├─> tak_messages.jsonl
                      ├─> tak_metrics.json
                      └─> EventBus (tak_message_sent, tak_bridge_completed)
```

### Components

- **cot.py** — CoT XML builder using `xml.etree.ElementTree`. Generates timestamped events with point/detail elements.
- **mapper.py** — Maps domain models (VehicleState, GeoObservation, MissionPlan, SystemEvent) to CoT XML strings.
- **transport.py** — `TakTransport` protocol with `DryRunTakTransport` (in-memory) and `UdpTakTransport` (socket).
- **bridge.py** — `TakBridge` orchestrator: configures transport, records messages, publishes events, writes JSONL.
- **metrics.py** — Message counts by category, type, and success/failure.

### Transport modes

| Mode | Description |
|------|-------------|
| `dry_run` | Validates and stores messages in memory. No network. Default for tests and CI. |
| `udp` | Sends UTF-8 XML datagrams to configured `cot_host:cot_port`. |

### Events

| Event | When |
|-------|------|
| `tak_message_sent` | Each CoT message successfully sent |
| `tak_message_failed` | Send failure with error details |
| `tak_bridge_completed` | Bridge closes with metrics summary |

### Future work

- **TLS TAK** — encrypted CoT transport for production deployments.
- **Authentication** — certificate-based TAK server auth.
- **Bidirectional commands** — receive ATAK commands back into the system.
- **Chat CoT** — send situational awareness text messages.
- **Mission packages** — structured data packages for ATAK.
- **Protobuf CoT** — more efficient binary encoding.
- **Streaming video** — real-time video feed integration.

## Integrated Pipeline

The integrated pipeline wires all subsystems into a single execution flow.

### Flow

```
Mission JSON
  → MissionPlanner → MissionPlan
  → MavlinkBridge (mock) → connect, upload, start, hold, return, land
  → PerceptionRunner → detections
  → TrackingRunner → tracks
  → GeolocalizationRunner → GeoObservations
  → TakBridge (dry_run) → CoT markers
  → MissionRecorder → events.jsonl
  → Summary + Report → pipeline_summary.json, report.md
```

### Components

- **IntegratedMissionPipeline** — orchestrates all subsystems, handles errors with `pipeline_failed` events.
- **summary.py** — builds structured pipeline summary dict and JSON.
- **report.py** — generates Markdown report with all sections.

### Error handling

If any phase raises an exception, the pipeline publishes `pipeline_failed` and re-raises. Resources (recorder files) are closed in `finally` blocks.

### Events

| Event | When |
|-------|------|
| `pipeline_started` | Pipeline begins |
| `pipeline_mission_planned` | MissionPlan created |
| `pipeline_completed` | All phases finished successfully |
| `pipeline_failed` | Unhandled exception in any phase |

## Edge Agent Runtime

The Edge Agent Runtime is the master operational orchestrator — a single process that coordinates all subsystems during a mission. It is designed to represent what would run on a Jetson or companion computer in a real deployment.

### Difference from Integrated Pipeline

The `IntegratedMissionPipeline` is a linear data-flow orchestrator focused on the processing pipeline. The Edge Agent Runtime adds:

- **Lifecycle state machine** with auditable transitions and structured events.
- **Context management** for all subsystem references with ordered, idempotent cleanup.
- **Health snapshots** for runtime monitoring.
- **PX4 SITL integration** with local mission generation, waypoint validation, and safe-land on failure.
- **Mode-aware execution** — different behavior for `mock` vs `px4_sitl`.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `modes.py` | `EdgeAgentMode` enum + validation | `mock` or `px4_sitl` |
| `context.py` | `EdgeAgentContext` | Mutable container for all subsystem references |
| `lifecycle.py` | `EdgeAgentLifecycle` | State machine with auditable transitions |
| `health.py` | Snapshot builder/writer | Runtime health JSON |
| `runtime.py` | `EdgeAgentRuntime` | Main orchestrator — phases A through H |

### Lifecycle States

```
CREATED → CONFIG_LOADED → RECORDER_STARTED → MISSION_PLANNED
→ VEHICLE_CONNECTED → MISSION_UPLOADED → MISSION_STARTED
→ PERCEPTION_RUNNING → TRACKING_RUNNING → GEOLOCALIZATION_DONE
→ TAK_PUBLISHED → REPORT_GENERATED → COMPLETED
```

Terminal states: `COMPLETED`, `FAILED`, `ABORTED`.

Every transition publishes `edge_agent_state_changed` to the EventBus. Terminal states publish their own events (`edge_agent_completed`, `edge_agent_failed`, `edge_agent_aborted`).

### Execution Phases

| Phase | State | Description |
|-------|-------|-------------|
| A. Initialization | `CONFIG_LOADED` → `RECORDER_STARTED` | Create EventBus, recorder, context, lifecycle |
| B. Planning | `MISSION_PLANNED` | Plan mission (local SITL or JSON-based) |
| C. MAVLink | `VEHICLE_CONNECTED` → `MISSION_STARTED` | Connect, validate, upload, start mission |
| D. Perception | `PERCEPTION_RUNNING` | Run perception + tracking pipeline |
| E. Geolocalization | `GEOLOCALIZATION_DONE` | Convert tracks to geo observations |
| F. TAK/C2 | `TAK_PUBLISHED` | Publish CoT markers for vehicle, mission, observations |
| G. Report | `REPORT_GENERATED` | Write summary, report, health snapshot |
| H. Post-mission | `COMPLETED` | Return home, land, close resources |

### Safety Architecture

- Default mode is `mock` — no hardware, no PX4, no network traffic.
- TAK is `dry_run` by default — no UDP datagrams.
- PX4 SITL requires explicit `--mode px4_sitl` and `SIMULATION_MODE` in config.
- Waypoints are validated against home position before upload.
- On failure, attempts safe return-home and land if PX4 SITL is active.
- No ATAK-to-drone command processing (receive only).
- No control of physical UAV hardware.

### Future: Jetson Deployment

- Replace mock backends with real sensor feeds.
- Use MAVSDK backend with real autopilot.
- Enable UDP TAK transport for ATAK integration.
- Add health monitoring, watchdog, and graceful degradation.

## Quality and Auditability

The system produces a complete audit trail through the combination of EventBus, MissionRecorder, and per-module artifacts.

### Audit chain

```
EventBus (pub/sub)
  → MissionRecorder (subscribes to "*")
  → events.jsonl (every SystemEvent in order)

Per-module artifacts:
  detections.jsonl    — perception
  tracks.jsonl        — tracker
  geo_observations.jsonl — geolocalizer
  tak_messages.jsonl  — TAK bridge

Aggregation:
  pipeline_summary.json — structured summary of all phases
  report.md             — human-readable mission report
```

### Quality gate

The quality gate validates pipeline outputs:

- **JSONL validation** — every line in every `.jsonl` file is valid JSON.
- **Artifact completeness** — all expected files exist in the run directory.
- **JSON validation** — every `.json` file parses correctly.
- **Report check** — `report.md` exists and is non-empty.
- **Summary structure** — `pipeline_summary.json` contains required keys.
- **Smoke checks** — project files exist, core imports succeed.

See `docs/quality_gate.md` for details.

## Tactical Replay Dashboard

The dashboard provides a local web UI for post-mission replay, situational awareness, and audit.

### Data flow

```
runs/<run_id>/
  *.json, *.jsonl, report.md
      │
      ▼
  loader.py  ──>  dict of artifacts
      │
      ├──>  timeline.py  ──>  sorted list of TimelineEvent
      ├──>  map_layers.py ──>  MapLayers (waypoints, observations, UAV, safety)
      │
      ▼
  app.py (FastAPI)
      │
      ├──>  REST API (/api/runs, /api/runs/{id}/timeline, /api/runs/{id}/map, ...)
      ├──>  WebSocket (/ws/live — periodic latest-run summary)
      └──>  Static HTML/JS/CSS (Leaflet map, timeline panel, metrics)
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `loader` | `sentinel/dashboard/loader.py` | Reads JSON/JSONL artifacts from run directories |
| `timeline` | `sentinel/dashboard/timeline.py` | Merges events, operator decisions, safety actions into sorted timeline |
| `map_layers` | `sentinel/dashboard/map_layers.py` | Extracts waypoints, observations, UAV positions, safety events for Leaflet |
| `models` | `sentinel/dashboard/models.py` | Pydantic models: RunInfo, TimelineEvent, MapLayers |
| `app` | `sentinel/dashboard/app.py` | FastAPI application with REST + WebSocket endpoints |
| `live` | `sentinel/dashboard/live.py` | Utility for monitoring latest run updates |
| `run_dashboard` | `apps/dashboard/run_dashboard.py` | CLI entry point (uvicorn launcher) |

### Safety

- **Read-only** — no commands are sent to the vehicle from the dashboard.
- **No authentication** — intended for local use in lab/simulation environments.
- **No ATAK input** — does not receive or process external commands.
- **No hardware control** — dashboard only reads artifact files.

### Use cases

- **Demo** — show mission results to stakeholders in a browser.
- **Post-mission review** — inspect timeline, map, metrics, and report after a run.
- **Auditability** — verify operator decisions, safety actions, and detection chain.
- **Debug** — visualize tracker output, geolocalization accuracy, and event sequencing.

## Demo Scenario Pack

The demo scenario pack provides reproducible, named mission scenarios for testing and demonstration. Each scenario runs the full pipeline with specific conditions and produces a complete artifact set.

### Data flow

```
DemoScenario (catalog)
    │
    ▼
DemoScenarioRunner
    │
    ├── AppConfig (YAML)
    ├── EventBus
    ├── MissionRecorder → events.jsonl
    │
    ├── PerceptionBackend (mock)
    ├── TrackingRunner
    ├── GeolocalizationRunner
    ├── TakBridge (dry_run)
    ├── OperatorDecisionGate + Simulator
    ├── SafetyActionExecutor + MockMavlinkBridge
    │
    ▼
RealTimeMissionLoop
    │
    └── runs/<timestamp>_demo_<scenario_id>/
          scenario_metadata.json
          events.jsonl, detections.jsonl, tracks.jsonl
          geo_observations.jsonl, tak_messages.jsonl
          operator_decisions.jsonl, safety_actions.jsonl
          realtime_metrics.json
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `DemoScenario` | `sentinel/demo/scenarios.py` | Pydantic model: scenario definition with parameters |
| `catalog` | `sentinel/demo/catalog.py` | Registry of 5 named scenarios |
| `runner` | `sentinel/demo/runner.py` | Orchestrates scenario execution end-to-end |
| `run_demo_scenario` | `apps/tools/run_demo_scenario.py` | CLI: list, run, run-all-safe |

### Scenarios

| ID | Trigger | Expected safety action |
|----|---------|----------------------|
| `observation_confirmed` | Operator confirms | TAK publishes confirmed observation |
| `operator_abort` | ABORT_MISSION decision | HOLD + LAND |
| `low_battery_return` | Battery < 20% | RETURN_HOME |
| `link_loss_return` | Link lost | RETURN_HOME |
| `px4_sitl_live_observation` | PX4 SITL | Live observation (requires PX4) |

### Safety

- All non-PX4 scenarios use MockMavlinkBackend — no hardware.
- Safety triggers are pre-injected before the loop starts.
- PX4 scenario requires explicit `--mode px4_sitl` and running PX4 SITL.
- No weapon, payload, or offensive capability in any scenario.

## Mission Evidence Package

The evidence package builder generates a portable, checksummed bundle from run artifacts for audit, replay, and demonstration.

### Data flow

```
runs/<run_id>/
  *.json, *.jsonl, report.md, *.mp4
      │
      ▼
  package_builder.py
      │
      ├──  summaries.py     ──>  summary.json, metrics_summary.json,
      │                         operator_summary.json, safety_summary.json
      ├──  manifest.py       ──>  replay_manifest.json (SHA-256 hashes, sizes)
      │                       ──>  artifacts_index.json (categorized inventory)
      │                       ──>  timeline.json (consolidated chronological)
      ├──  checksums.py      ──>  checksums.json (package file hashes)
      │
      ▼
  mission_package/
    summary.json, metrics_summary.json, operator_summary.json,
    safety_summary.json, timeline.json, replay_manifest.json,
    artifacts_index.json, checksums.json

  exporter.py ──> mission_package.zip (optional)
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `checksums` | `sentinel/evidence/checksums.py` | SHA-256 computation for run artifacts |
| `summaries` | `sentinel/evidence/summaries.py` | Aggregated metrics, operator, safety, and run summaries |
| `manifest` | `sentinel/evidence/manifest.py` | Replay manifest, artifacts index, consolidated timeline |
| `package_builder` | `sentinel/evidence/package_builder.py` | Orchestrates all evidence generation into mission_package/ |
| `exporter` | `sentinel/evidence/exporter.py` | Creates mission_package.zip from evidence directory |

### Integration

The demo scenario runner calls `build_evidence_package(run_dir)` automatically after each scenario completes. The edge agent runtime and integrated pipeline can also call it directly.

### Package contents

| File | Description |
|------|-------------|
| `summary.json` | Run metadata, scenario info, artifact line counts |
| `metrics_summary.json` | All per-module metrics JSONs aggregated |
| `operator_summary.json` | Decision counts by action and operator |
| `safety_summary.json` | Safety action counts by trigger/action, success flag |
| `timeline.json` | Merged chronological timeline from all event JSONLs |
| `replay_manifest.json` | Full artifact inventory with SHA-256 hashes, sizes |
| `artifacts_index.json` | Artifacts grouped by subsystem category |
| `checksums.json` | SHA-256 of all files inside the package |

## Sensor Ingestion

The sensor ingestion module provides unified video source abstraction supporting local files, USB webcams, and RTSP streams with health monitoring and automatic reconnection.

### Data flow

```
VideoSourceConfig (YAML)
    │
    ▼
  factory.py ──> ManagedVideoSource
    │
    ├─ source_type=file    ──> cv2.VideoCapture(file_path)
    ├─ source_type=webcam  ──> cv2.VideoCapture(index) + resolution
    └─ source_type=rtsp   ──> cv2.VideoCapture(rtsp_url) + reconnect
    │
    ├── read_frame() ──> (success, frame)
    ├── health ──> SensorHealth (fps, dropped, reconnects, stale)
    ├── write_health_json() ──> sensor_health.json
    └── events ──> sensor_connected, sensor_disconnected, sensor_reconnected
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `types` | `sentinel/sensors/types.py` | `VideoSourceType` enum, `SensorHealth` dataclass |
| `managed_source` | `sentinel/sensors/managed_source.py` | `ManagedVideoSource` — unified capture with health and reconnect |
| `factory` | `sentinel/sensors/factory.py` | Creates `ManagedVideoSource` from `VideoSourceConfig` |

### Source types

| Type | Open | Reconnect | EOF behavior |
|------|------|-----------|--------------|
| `file` | `cv2.VideoCapture(path)` | No | Returns `(False, None)` — loop ends |
| `webcam` | `cv2.VideoCapture(index)` | Yes (if enabled) | Returns `(False, None)` after failed reconnect |
| `rtsp` | `cv2.VideoCapture(url)` | Yes (if enabled) | Returns `(False, None)` after failed reconnect |

### Health monitoring

`SensorHealth` tracks: `connected`, `fps_estimate` (deque of 30 timestamps), `dropped_frames`, `reconnect_count`, `last_frame_utc`, `stale` (>2x target period without frame).

Persisted to `sensor_health.json` in the run directory when using ManagedVideoSource.

### Events

| Event | When |
|-------|------|
| `sensor_connected` | Source opened successfully |
| `sensor_disconnected` | Read failure on webcam/RTSP |
| `sensor_reconnected` | Reconnection succeeded |

### Reconnect logic (RTSP/webcam)

1. `read_frame()` returns `False` from `cap.read()`.
2. Increment `dropped_frames`. Publish `sensor_disconnected`.
3. Release old capture. Sleep `reconnect_interval_s`.
4. Call `open()` again. If success, increment `reconnect_count`, publish `sensor_reconnected`.
5. Return to frame reading. If reconnect fails, return `(False, None)`.

### Integration with RealTimeMissionLoop

`RealTimeMissionLoop` accepts an optional `video_source_provider: ManagedVideoSource`. When provided, the loop uses `provider.read_frame()` instead of raw `cv2.VideoCapture`. Sensor health metrics (`dropped_frames`, `capture_fps`, `reconnect_count`) are included in `realtime_metrics.json`.

When not provided (default), the loop uses the legacy `_open_video_source()` path — fully backward compatible.

### Limitations

- No CSI camera (Jetson-specific MIPI).
- No GStreamer pipelines.
- No CUDA/TensorRT optimization.
- No audio processing.
- RTSP reconnect blocks the loop during retry interval.
- No adaptive bitrate or stream quality control.

## UAV Telemetry Layer

The UAV telemetry layer records vehicle state and visualizes it on the dashboard map.

### Data flow

```
MockMovementTelemetryProvider / MavlinkTelemetryProvider
    │
    ▼ (per frame)
VehicleState
    │
    ├── _vehicle_state_to_record() ──> vehicle_states.jsonl (one per frame)
    ├── EventBus (every 5 frames)  ──> vehicle_state_updated event
    │
    ▼
Dashboard:
  loader.py       ──> reads vehicle_states.jsonl
  map_layers.py   ──> extracts uav_positions + uav_track
  live.py         ──> tails vehicle_states.jsonl, builds latest_uav_state + uav_track_tail
  app.js          ──> renders UAV icon (heading-rotated), polyline track, telemetry panel
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `MockMovementTelemetryProvider` | `sentinel/realtime/mock_telemetry.py` | Simulated UAV movement along waypoints |
| `MavlinkTelemetryProvider` | `sentinel/realtime/telemetry_stream.py` | Real MAVLink telemetry with stale fallback |
| `_vehicle_state_to_record` | `sentinel/realtime/loop.py` | Converts VehicleState to JSONL record |
| `map_layers` | `sentinel/dashboard/map_layers.py` | Extracts UAV positions, track, telemetry from artifacts |
| `live` | `sentinel/dashboard/live.py` | Tails vehicle_states.jsonl, builds live UAV payload |

### Mock movement

`MockMovementTelemetryProvider` interpolates between waypoints using haversine distance and bearing. Each tick advances the UAV along the path at the configured speed. Battery decreases by 0.02% per frame. Default waypoints form a small triangle near the base position.

### Vehicle state record

Each frame writes to `vehicle_states.jsonl`:
```json
{
  "timestamp_utc": "...",
  "vehicle_id": "uav_001",
  "position": {"lat": ..., "lon": ..., "alt_m": ...},
  "heading_deg": ...,
  "groundspeed_mps": ...,
  "battery_pct": ...,
  "mode": "AUTO",
  "armed": true,
  "stale": false
}
```

### Dashboard visualization

- **UAV icon**: Red triangle (`divIcon`) rotated via CSS `transform: rotate(heading)`.
- **Track polyline**: Yellow line connecting all vehicle state positions.
- **Telemetry panel**: Table showing altitude, speed, heading, battery, mode, armed, last update, stale flag.
- **Live follow**: In live mode, map pans to follow the UAV position.

### Events

| Event | When |
|-------|------|
| `vehicle_state_updated` | Every 5 frames during the realtime loop |

### Map layer model

`UavPositionLayer` now includes: `alt_m`, `speed_mps`, `battery_pct`, `mode`, `armed`, `stale`, `vehicle_id`.

`UavTrackPoint`: `lat`, `lon`, `alt_m`, `timestamp_utc`.

`MapLayers` includes `uav_track: list[UavTrackPoint]`.

### WebSocket live payload

The live WebSocket payload now includes:
- `latest_uav_state`: Full telemetry dict from the latest vehicle state
- `uav_track_tail`: Last 200 track points for polyline rendering

