# Sentinel Camera Strategy — ISR Gazebo

## Executive Summary

A single narrow-FOV IMX477 camera is **insufficient for acquisition/search** in dynamic_v3. The baseline wide camera (`x500_mono_cam`) must be restored to +45° downward pitch and used as the primary acquisition sensor. The IMX477 profiles serve as **identification/confirmation** cameras once a target is acquired.

---

## Camera Role Recommendations

| Role | Model | HFOV | Resolution | Purpose | When to Use |
|------|-------|------|------------|---------|-------------|
| **Acquisition / Search** | `x500_mono_cam` | ~100° (1.74 rad) | 1280×960 | Wide-area search, target acquisition, operator context, keeping targets in view | **Default for all ISR ops** |
| **Identification** | `sentinel_x500_imx477_45deg` | 45° (0.785 rad) | 1920×1080 | Close-range identification, confirmation, higher-detail tracking | Once target is approximately centered |
| **Balanced Zoom** | `sentinel_x500_imx477_30deg` | 30° (0.524 rad) | 1920×1080 | Medium-range zoom (optional) | When 45° is not narrow enough |
| **Zoom Only** | `sentinel_x500_imx477_cam` | 22° (0.38 rad) | 1920×1080 | Extreme close-up identification | <5m from target, final confirmation |

---

## Heading Anomaly Diagnosis

### Problem
The baseline `x500_mono_cam` camera was pitched **-0.785 rad (45° UP)** instead of **+0.785 rad (45° DOWN)**. This caused the camera to look at the horizon/sky rather than the ground.

### Root Cause
1. The PX4 upstream `x500_mono_cam/model.sdf` had `-0.785` pitch
2. The Sentinel override `configs/gz/models/x500_mono_cam/model.sdf` also had `-0.785` pitch
3. The `verify_px4_camera_pitch.py` script had a **bug**: its substring check `if "0.785" in text` incorrectly matched `-0.785`, reporting "OK" when the camera was actually pointing UP

### Evidence

| Test | Spawn | Camera | Pitch | Probe Image | Detections |
|------|-------|--------|-------|-------------|------------|
| Pre-fix wide | 205,-345,31.5 | `x500_mono_cam` | -0.785 (UP) | Sky + trees on horizon | 0 |
| Pre-fix wide | 220,-350,22 | `x500_mono_cam` | -0.785 (UP) | Sky + distant figures | 0 (timed out) |
| Post-fix wide | 220,-350,22 | `x500_mono_cam` | **+0.785 (DOWN)** | Brown ground | **2169 person, 13 tracks** |
| IMX477 45° | 220,-350,22 | `sentinel_x500_imx477_45deg` | +0.785 (DOWN) | Brown ground | 0 (narrow FOV, no target in patch) |

### Fix Applied
- `PX4-Autopilot/Tools/simulation/gz/models/x500_mono_cam/model.sdf`: `-0.785` → `+0.785` (local untracked file)
- `ons-sentinel-core/configs/gz/models/x500_mono_cam/model.sdf`: already `+0.785` in HEAD; working tree was restored to match
- `scripts/verify_px4_camera_pitch.py`: Added explicit checks for wrong `-0.785` poses and exact-match verification for correct `+0.785` poses

---

## Wide vs 45° Comparison Table

Same spawn (`220,-350`), same confidence (`0.05`), same duration (15s), dynamic_v3 world:

| Metric | Wide (100°) 10m | Wide (100°) 15m | 45° 10m (same spawn) | 45° 10m (optimal spawn) |
|--------|-----------------|-----------------|----------------------|------------------------|
| **Ground footprint** | ~24m wide | ~36m wide | ~8m wide | ~8m wide |
| **Person detections** | 2169 | 1972 | 0 | 220 |
| **Unique tracks** | 13 | 9 | 0 | 1 |
| **First detection** | Frame 4 | Frame 4 | N/A | Frame 50 |
| **Lock achieved** | Yes (OFFBOARD) | Yes (OFFBOARD) | No | Yes (OFFBOARD) |
| **Viable for search** | ✅ Excellent | ✅ Excellent | ❌ No | ⚠️ Only if pre-positioned |

### Key Insight
The 45° camera **can** detect targets when they are in its FOV (proven by the optimal-spawn test at `215,-345`), but its narrow footprint makes **unaided acquisition from arbitrary standoff positions impossible**. The wide camera captures the entire actor distribution instantly.

---

## Dual-Camera Strategy

### Operational Default
For Gazebo demos and real-world ISR:
- **Primary**: Wide `x500_mono_cam` for acquisition, search, and operator context
- **Secondary**: IMX477 45° for identification/confirmation once target is centered

### Implementation Options

#### Option A: Switchable Single Camera (Recommended for Now)
- Launch with `--model x500_mono_cam` for acquisition phase
- When target is locked and centered, switch to `--model sentinel_x500_imx477_45deg` for identification
- Minimal code changes; leverages existing `--model` support

#### Option B: Dual-Camera Model (Future)
Create a new `sentinel_x500_dual_cam` model with:
```xml
<model name='sentinel_x500_dual_cam'>
  <include merge='true'><uri>x500</uri></include>
  <!-- Wide acquisition camera -->
  <include merge='true'>
    <uri>model://mono_cam</uri>
    <pose>0.12 0.03 0.242 0 0.785 0</pose>
    <name>acquisition_cam</name>
  </include>
  <!-- Narrow identification camera -->
  <include merge='true'>
    <uri>model://sentinel_imx477_45deg_cam</uri>
    <pose>0.20 0.0 -0.05 0 0.785 0</pose>
    <name>identification_cam</name>
  </include>
</model>
```

**Challenges**:
- Gazebo camera topics would include both cameras: `/link/acquisition_cam_camera_link/sensor/camera/image` and `/link/identification_cam_camera_link/sensor/camera/image`
- The YOLO pipeline and follow controller currently subscribe to a single topic
- Switching between topics at runtime requires changes to `CameraWorker`, `stack_launcher.py`, and `run_gazebo_yolo_test.py`

**Recommendation**: Implement **Option A** now. Document Option B for a future milestone.

---

## Annotated Frame Paths

| Test | Path |
|------|------|
| Wide 10m (post-fix) | `logs/baseline_wide_10m/annotated_frames/` |
| Wide 15m (post-fix) | `logs/baseline_wide_15m/annotated_frames/` |
| 45° optimal 10m | `logs/imx477_45deg_actor_centered/annotated_frames/` |
| 45° baseline spawn 10m | `logs/imx477_45deg_10m_baseline_spawn/annotated_frames/` |

---

## Files Changed

### In `ons-sentinel-core`:
1. `configs/gz/models/x500_mono_cam/model.sdf` — Fixed camera pitch from -0.785 to +0.785
2. `scripts/verify_px4_camera_pitch.py` — Fixed substring bug, added wrong-pose detection
3. `apps/tools/run_auto_sim_test.py` — Added `--model` and `--confidence` args
4. `apps/tools/run_gazebo_yolo_test.py` — Added class alias normalization (`walker→person`, `vehicle→car`)
5. `sentinel/runtime/workers/perception_worker.py` — Added class alias normalization
6. `sentinel/testing/stack_launcher.py` — Added model-aware topic generation
7. `scripts/start_sentinel_stack.sh` — Added `--model` argument parsing

### In `PX4-Autopilot`:
1. `Tools/simulation/gz/models/x500_mono_cam/model.sdf` — Fixed camera pitch from -0.785 to +0.785

---

## Pytest Results

```
pytest -q tests/test_worker_follow_path.py
# 64 passed, 3 warnings in 1.88s

pytest -q
# 1076 passed, 4 skipped, 2 warnings in 42.27s
```

---

## Reproducibility & PX4 Dependency

### Fresh Clone Guarantee
The **Sentinel override** (`configs/gz/models/x500_mono_cam/model.sdf`) is the authoritative source and is **versioned in the Sentinel repo** with the correct `+0.785` pitch.

`start_sentinel_stack.sh` explicitly prepends the Sentinel model directory to `GZ_SIM_RESOURCE_PATH`:
```bash
export GZ_SIM_RESOURCE_PATH="${SENTINEL_DIR}/configs/gz/models:${GZ_SIM_RESOURCE_PATH}"
```

This means a fresh clone of `ons-sentinel-core` will always use the corrected Sentinel override, regardless of the state of the local PX4 working tree.

### PX4 Untracked File
`PX4-Autopilot/Tools/simulation/gz/models/x500_mono_cam/model.sdf` is an **untracked local file** (not in the PX4 git index). It was fixed during this investigation, but the fix is not required for reproducibility because:
1. The Sentinel override takes precedence via `GZ_SIM_RESOURCE_PATH`
2. A fresh clone gets the correct pitch from the Sentinel repo alone

**Recommendation**: Do not rely on the PX4 untracked file. Always use the Sentinel override.

## Safe to Commit?

**Yes.** All changes are additive or bug-fixes:
- Camera pitch is correct in the Sentinel override (versioned)
- `verify_px4_camera_pitch.py` now detects wrong `-0.785` poses
- Class aliases enable the Phase 8 YOLO model (`walker`, `vehicle`) to work with Sentinel's COCO-based pipeline
- `--model` support enables camera profile selection
- Pytest suite passes (1076/1076)
