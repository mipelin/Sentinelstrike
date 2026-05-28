"""YOLO detection test against live Gazebo camera feed.

Usage:
    python -m apps.tools.run_gazebo_yolo_test \
        --topic /world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image \
        --max-frames 100

    # With specific model, device, and confidence:
    python -m apps.tools.run_gazebo_yolo_test \
        --topic ... --model-path yolov8n.pt --device cuda --confidence 0.25 --max-frames 200

    # With tracking (ByteTrack, tuned for aerial sim):
    python -m apps.tools.run_gazebo_yolo_test --topic ... --track bytetrack

    # With IoU tracker (stable IDs, smoothed boxes):
    python -m apps.tools.run_gazebo_yolo_test --topic ... --track iou

    # Mock backend (no YOLO required):
    python -m apps.tools.run_gazebo_yolo_test --topic ... --backend mock --max-frames 50

    # ISR tracker (persistent IDs, camera motion compensation):
    python -m apps.tools.run_gazebo_yolo_test --topic ... --track isr

    # ISR tracker with follow mode:
    python -m apps.tools.run_gazebo_yolo_test --topic ... --track isr --follow-track TGT-001
"""

from __future__ import annotations

import argparse
import inspect
import math
import signal
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import gz.transport13 as _gzt
    import gz.msgs10.image_pb2 as _image_pb2

    _GZ_AVAILABLE = True
except ImportError:
    _GZ_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ── Class alias mapping for custom YOLO models ─────────────────────────
_CLASS_ALIASES: dict[str, str] = {
    "walker": "person",
    "pedestrian": "person",
    "human": "person",
    "vehicle": "car",
    "automobile": "car",
    "auto": "car",
}


def _normalize_class_name(name: str) -> str:
    """Normalize a raw YOLO class name to Sentinel's canonical COCO name."""
    return _CLASS_ALIASES.get(name, name)


from apps.tools.view_gazebo_camera import FrameState
def _selectable_targets(detections: list[dict], frame_w: int) -> list[dict]:
    """Return sorted list of selectable tracks for F/Tab target selection.

    Priority: visible person > visible vehicle > nearest center > highest confidence.
    Only confirmed/active tracks with track_id are included.
    """
    VISIBLE = {"CONFIRMED", "ACTIVE", "active", "REACQUIRED"}
    result = []
    for d in detections:
        tid = d.get("track_id")
        if tid is None:
            continue
        status = d.get("status", "")
        if status == "LOST":
            continue
        bbox = d.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2.0
        dist_center = abs(cx - frame_w / 2.0)
        is_visible = status in VISIBLE
        cls = d.get("class", "")
        is_person = cls == "person"
        conf = d.get("confidence", 0.0)
        result.append({
            "track_id": tid,
            "class": cls,
            "confidence": conf,
            "dist_center": dist_center,
            "is_visible": is_visible,
            "is_person": is_person,
            "status": status,
        })
    result.sort(key=lambda t: (
        0 if t["is_visible"] else 1,
        0 if t["is_person"] else 1,
        t["dist_center"],
        -t["confidence"],
    ))
    return result


def _draw_target_hud(
    out: np.ndarray,
    selectable_count: int,
    selected_target: str | None,
    locked_target: str | None,
    follow_on: bool,
    autonomy_on: bool,
    follow_state: str,
    lost_time_s: float,
) -> None:
    """Draw a small HUD showing target selection state."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    x = out.shape[1] - 260
    lines = [
        f"selectable={selectable_count}",
        f"selected={selected_target or 'None'}",
        f"locked={locked_target or 'None'}",
        f"follow={'ON' if follow_on else 'OFF'}  autonomy={'ON' if autonomy_on else 'OFF'}",
        f"state={follow_state}  lost={lost_time_s:.1f}s",
    ]
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 2 and locked_target else (200, 200, 200)
        cv2.putText(out, line, (x, y + i * 18), font, 0.4, color, 1, cv2.LINE_AA)


def _check_airborne(flight_bridge, min_alt_m: float = 3.0) -> tuple[bool, str]:
    """Check if the vehicle is armed and airborne. Returns (ok, reason)."""
    if flight_bridge is None:
        return False, "Flight bridge not connected"
    tele = flight_bridge.telemetry_cache.snapshot()
    if not tele.connected:
        return False, "MAVSDK not connected"
    if not tele.armed:
        return False, "Vehicle DISARMED"
    if tele.position is None or tele.position.alt_m is None:
        return False, "Altitude unknown"
    if tele.position.alt_m < min_alt_m:
        return False, f"Altitude {tele.position.alt_m:.1f}m < {min_alt_m:.1f}m (landed?)"
    return True, ""


def _enable_autonomy_safe(flight_bridge, min_alt_m: float = 3.0) -> bool:
    """Enable autonomy only if flight preconditions are met."""
    ok, reason = _check_airborne(flight_bridge, min_alt_m)
    if not ok:
        print(f"[SAFETY] Follow/autonomy disabled: {reason}")
        return False
    flight_bridge.autonomy_enabled = True
    return True


def _draw_telemetry_hud(out: np.ndarray, flight_bridge) -> None:
    """Draw telemetry state (armed, alt, mode) in top-left corner."""
    if flight_bridge is None:
        return
    tele = flight_bridge.telemetry_cache.snapshot()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 10, 25
    lines = []
    if not tele.connected:
        lines.append(("MAVSDK: NOT CONNECTED", (0, 0, 255)))
    else:
        armed_str = "ARMED" if tele.armed else "DISARMED"
        armed_color = (0, 255, 0) if tele.armed else (0, 0, 255)
        lines.append((f"{armed_str}", armed_color))
        alt = tele.position.alt_m if tele.position else None
        if alt is not None:
            alt_color = (0, 255, 0) if alt >= 3.0 else (0, 0, 255)
            lines.append((f"ALT: {alt:.1f}m", alt_color))
        else:
            lines.append(("ALT: --", (0, 0, 255)))
        lines.append((f"MODE: {tele.mode}", (200, 200, 200)))
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (x, y + i * 20), font, 0.5, color, 1, cv2.LINE_AA)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _build_tracker_config(args: argparse.Namespace) -> str:
    """Build tracker YAML config path from CLI args."""
    project_cfg = Path("configs/tracking") / f"{args.track}_gazebo.yaml"
    has_overrides = any([
        args.track_buffer, args.match_thresh, args.track_high, args.track_low,
    ])

    if not has_overrides and project_cfg.exists():
        return str(project_cfg)

    # Load base config or defaults
    config: dict = {
        "tracker_type": args.track,
        "track_high_thresh": 0.35,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.4,
        "track_buffer": 60,
        "match_thresh": 0.7,
        "fuse_score": True,
    }

    if project_cfg.exists():
        import yaml

        with open(project_cfg) as f:
            config.update(yaml.safe_load(f))

    # Apply CLI overrides
    if args.track_high is not None:
        config["track_high_thresh"] = args.track_high
    if args.track_low is not None:
        config["track_low_thresh"] = args.track_low
    if args.track_buffer is not None:
        config["track_buffer"] = args.track_buffer
    if args.match_thresh is not None:
        config["match_thresh"] = args.match_thresh

    import yaml

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="tracker_")
    yaml.dump(config, tmp)
    tmp.close()
    return tmp.name


def _run_yolo_detection(
    model: object,
    frame: np.ndarray,
    confidence: float,
    classes: list[str],
    device: str,
    imgsz: int | None = None,
) -> tuple[list[dict], object]:
    """Run YOLO detection. Returns (detections, raw_results)."""
    kwargs: dict = {"conf": confidence, "verbose": False, "device": device}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    results = model(frame, **kwargs)
    return _extract_detections(results, classes)


def _run_mock_backend(frame: np.ndarray) -> list[dict]:
    h, w = frame.shape[:2]
    return [
        {
            "class": "person",
            "confidence": 0.85,
            "bbox": [w * 0.3, h * 0.3, w * 0.7, h * 0.8],
            "track_id": 1,
        },
        {
            "class": "car",
            "confidence": 0.72,
            "bbox": [w * 0.05, h * 0.5, w * 0.4, h * 0.9],
            "track_id": 2,
        },
    ]


def _extract_detections(results: list, classes: list[str]) -> tuple[list[dict], list]:
    """Extract detections from YOLO results (detect or track).

    Normalizes custom model class names (e.g. walker -> person) before
    filtering against the expected class list.
    """
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_name = r.names[cls_id]
            norm_name = _normalize_class_name(raw_name)
            if classes and norm_name not in classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            track_id = None
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id[0])
            detections.append({
                "class": norm_name,
                "raw_class": raw_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "track_id": track_id,
            })
    return detections, results


# --- IoU tracker integration ---

def _yolo_dets_to_tracker_dets(yolo_dets: list[dict], frame_id: int) -> list:
    """Convert YOLO detection dicts to sentinel Detection objects."""
    from sentinel.common.types import BoundingBox, Detection

    timestamp = datetime.now(timezone.utc).isoformat()
    result = []
    for i, d in enumerate(yolo_dets):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        result.append(Detection(
            frame_id=frame_id,
            timestamp_utc=timestamp,
            class_name=d["class"],
            confidence=d["confidence"],
            bbox_xyxy=BoundingBox(
                x1=min(x1, x2 - 1), y1=min(y1, y2 - 1),
                x2=max(x2, x1 + 1), y2=max(y2, y1 + 1),
            ),
            source="yolo_gazebo",
            detection_id=f"det_{frame_id}_{i}",
        ))
    return result


def _tracker_tracks_to_dets(tracks: list) -> list[dict]:
    """Convert sentinel Track objects to overlay-friendly dicts with smoothed bboxes."""
    result = []
    for t in tracks:
        if t.bbox_xyxy is None:
            continue
        meta = t.metadata or {}
        result.append({
            "class": t.class_name,
            "confidence": meta.get("smoothed_confidence", t.confidence),
            "bbox": [t.bbox_xyxy.x1, t.bbox_xyxy.y1, t.bbox_xyxy.x2, t.bbox_xyxy.y2],
            "track_id": t.track_id,
            "status": t.status,
            "age_frames": t.age_frames,
            "lost_frames": t.lost_frames,
            "reacquired_count": meta.get("reacquired_count", 0),
            "smoothed_bbox": meta.get("track_quality_score", 0) > 0,
        })
    return result


def _safe_rect(
    img: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    color: tuple | list | np.ndarray,
    thickness: int = cv2.FILLED,
) -> bool:
    """Draw a cv2.rectangle with sanitized coordinates and bounds clamping.

    Returns False and skips drawing if the rectangle is degenerate or
    fully outside the image.
    """
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    h, w = img.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return False

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))

    c = tuple(int(v) for v in color)
    cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)
    return True


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
) -> None:
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for i in range(x1, x2, dash_len * 2):
        cv2.line(img, (i, y1), (min(i + dash_len, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i + dash_len, x2), y2), color, thickness)
    for i in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_len, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i + dash_len, y2)), color, thickness)


def _draw_overlay(
    frame: np.ndarray,
    detections: list[dict],
    capture_fps: float,
    process_fps: float,
    inference_ms: float,
    device: str,
    frame_id: int,
    tracker_type: str | None = None,
    unique_track_count: int = 0,
    active_track_count: int = 0,
    lost_track_count: int = 0,
    reacquired_count: int = 0,
) -> np.ndarray:
    """Draw bounding boxes, labels, track IDs, and FPS on frame."""
    out = frame.copy()

    class_colors = {
        "person": (0, 255, 0),
        "car": (0, 255, 255),
        "truck": (255, 165, 0),
        "bus": (0, 165, 255),
    }
    status_colors = {
        "ACTIVE": (0, 255, 0),       # green
        "active": (0, 255, 0),        # green (lowercase from ByteTrack/IoU)
        "LOST": (0, 0, 255),          # red
        "lost": (0, 0, 255),
        "REACQUIRED": (255, 255, 0),   # cyan
        "TENTATIVE": (0, 255, 255),    # yellow
        "tentative": (0, 255, 255),
    }
    default_color = (0, 200, 200)

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        status = d.get("status", "active")
        if d.get("track_id") is not None:
            color = status_colors.get(status, class_colors.get(d["class"], default_color))
        else:
            color = class_colors.get(d["class"], default_color)

        # Dashed box for LOST tracks, solid for everything else
        if status in ("LOST", "lost"):
            _draw_dashed_rect(out, (x1, y1), (x2, y2), color, 2)
        else:
            _safe_rect(out, x1, y1, x2, y2, color, 2)

        label_parts = [d["class"], f"{d['confidence']:.2f}"]
        if d.get("track_id") is not None:
            label_parts.append(str(d["track_id"]))
        if status in ("LOST", "lost"):
            label_parts.append("LOST")
        elif status in ("REACQUIRED",):
            label_parts.append("REACQ")
        elif status in ("TENTATIVE", "tentative") and d.get("track_id") is not None:
            label_parts.append("?")
        label = " ".join(label_parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        _safe_rect(out, x1, y1 - th - 8, x1 + tw + 4, y1, color)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Two-line HUD bar
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), cv2.FILLED)
    line1 = f"Cap:{capture_fps:.0f} Proc:{process_fps:.1f} Inf:{inference_ms:.0f}ms Dev:{device} Dets:{len(detections)}"
    cv2.putText(out, line1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    line2 = f"Frm:{frame_id}"
    if tracker_type == "persistent":
        line2 += f" Act:{active_track_count} Lost:{lost_track_count} Reaq:{reacquired_count} Uniq:{unique_track_count}"
    elif tracker_type:
        line2 += f" Active:{active_track_count} Unique:{unique_track_count} Trk:{tracker_type}"
    cv2.putText(out, line2, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)

    return out


def _draw_isr_overlay(
    frame: np.ndarray,
    detections: list[dict],
    capture_fps: float,
    process_fps: float,
    inference_ms: float,
    device: str,
    frame_id: int,
    unique_track_count: int = 0,
    active_track_count: int = 0,
    lost_track_count: int = 0,
    reacquired_count: int = 0,
    follow_id: str | None = None,
    follow_diag: object = None,
    debug_world_projection: bool = False,
) -> np.ndarray:
    """Draw ISR-style overlay with trajectories, velocity vectors, heading, status colors."""
    out = frame.copy()
    h, w = out.shape[:2]

    status_colors = {
        "CONFIRMED": (0, 220, 0),    # green
        "LOST": (0, 0, 255),          # red
        "TENTATIVE": (0, 255, 255),   # yellow
    }
    follow_state_colors = {
        "LOCKED": (0, 255, 0),
        "SEARCHING": (0, 255, 255),
        "LOST": (0, 0, 255),
        "REACQUIRED": (255, 255, 0),
        "CANDIDATE": (0, 165, 255),
        "UNLOCKED": (150, 150, 150),
    }
    class_colors = {
        "person": (0, 255, 0),
        "car": (0, 255, 255),
        "truck": (255, 165, 0),
        "bus": (0, 165, 255),
    }

    follow_target = None

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        status = d.get("status", "TENTATIVE")
        tid = str(d.get("track_id", "?"))
        is_followed = follow_id is not None and tid == follow_id

        if is_followed:
            follow_target = d

        # Dim non-followed targets when in follow mode
        if follow_id and not is_followed:
            color = (60, 60, 60)
        else:
            color = status_colors.get(status, (0, 200, 200))

        # Recovery flash: cyan for recently recovered tracks
        recovered = d.get("recovered", False)
        if recovered:
            flash = d.get("recovered_frames_ago", 0)
            if flash > 0:
                alpha = flash / 10.0
                color = (
                    int(color[0] * (1 - alpha) + 255 * alpha),
                    int(color[1] * (1 - alpha) + 255 * alpha),
                    int(color[2] * (1 - alpha) + 0 * alpha),
                )

        # Draw trajectory tail
        traj = d.get("trajectory", [])
        if len(traj) > 2 and not (follow_id and not is_followed):
            pts = np.array([(int(p[0]), int(p[1])) for p in traj[-50:]], dtype=np.int32)
            n_pts = len(pts)
            for i in range(1, n_pts):
                frac = i / n_pts
                tail_color = (
                    int(color[0] * frac * 0.6),
                    int(color[1] * frac * 0.6),
                    int(color[2] * frac * 0.6),
                )
                cv2.line(out, pts[i - 1], pts[i], tail_color, 1, cv2.LINE_AA)

        # Draw bounding box
        thickness = 3 if is_followed else 2
        if status == "LOST":
            _draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness)
        else:
            _safe_rect(out, x1, y1, x2, y2, color, thickness)

        # Velocity vector
        vel = d.get("velocity", [0, 0])
        speed = d.get("speed", 0)
        if speed > 1.0 and not (follow_id and not is_followed):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            scale = min(speed * 3, 60)
            vx, vy = vel
            mag = max(math.sqrt(vx * vx + vy * vy), 0.01)
            end_x = int(cx + vx / mag * scale)
            end_y = int(cy + vy / mag * scale)
            cv2.arrowedLine(out, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

        # Label
        label_parts = [tid, d["class"]]
        if status == "LOST":
            label_parts.append("LOST")
        elif recovered:
            label_parts.append("RECOV")
        label = " ".join(label_parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        _safe_rect(out, x1, y1 - th - 10, x1 + tw + 6, y1, color)
        cv2.putText(out, label, (x1 + 3, y1 - 4), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Identity descriptor line below box
        age = d.get("age_frames", 0)
        identity = d.get("identity_descriptor", "")
        stability = d.get("identity_stability", 0.0)
        age_text = f"{age}f spd:{speed:.0f}"
        if identity:
            age_text += f" [{identity}]"
        if stability > 0:
            stab_char = "S" if stability > 0.7 else ("M" if stability > 0.4 else "U")
            age_text += f" {stab_char}:{stability:.0%}"
        cv2.putText(out, age_text, (x1, y2 + 14), font, 0.35, color, 1, cv2.LINE_AA)

        # Predicted position for lost tracks
        if status == "LOST":
            pred = d.get("predicted_position")
            if pred:
                px, py = int(pred[0]), int(pred[1])
                cv2.drawMarker(out, (px, py), (0, 200, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(out, "PRED", (px + 8, py - 8), font, 0.3, (0, 200, 255), 1, cv2.LINE_AA)

        # Occlusion state indicator
        occ = d.get("occlusion_state", "")
        if occ and occ != "VISIBLE":
            occ_colors = {
                "OCCLUDED": (0, 0, 255),
                "SEARCHING": (0, 255, 255),
                "REACQUIRED": (0, 255, 0),
                "STALE": (128, 128, 128),
            }
            occ_color = occ_colors.get(occ, (200, 200, 200))
            cv2.putText(out, occ, (x2 + 4, y1 + 12), font, 0.35, occ_color, 1, cv2.LINE_AA)
            occ_frames = d.get("occluded_frames", 0)
            if occ_frames > 0:
                cv2.putText(out, f"{occ_frames}f", (x2 + 4, y1 + 26), font, 0.3, occ_color, 1, cv2.LINE_AA)

        # World-space info line
        wp = d.get("world_position")
        if wp:
            wv = d.get("world_velocity", {})
            speed_mps = wv.get("speed_mps", 0) if wv else 0
            heading_deg = wv.get("heading_deg", 0) if wv else 0
            world_text = f"{wp.get('lat', 0):.5f},{wp.get('lon', 0):.5f} d:{wp.get('ground_distance_m', 0):.0f}m"
            if speed_mps > 0.1:
                world_text += f" v:{speed_mps:.1f}m/s h:{heading_deg:.0f}"
            cv2.putText(out, world_text, (x1, y2 + 28), font, 0.3, (180, 180, 255), 1, cv2.LINE_AA)

        # World projection debug status
        if debug_world_projection:
            proj_status = d.get("projection_status", "NEVER_ATTEMPTED")
            status_color = (0, 255, 0) if proj_status == "OK" else (0, 0, 255)
            status_text = f"WORLD: {proj_status}"
            cv2.putText(out, status_text, (x1, y2 + 42), font, 0.3, status_color, 1, cv2.LINE_AA)

        # Hypothesis probability bar
        hyps = d.get("hypotheses", [])
        if hyps:
            hyp_colors = {
                "OCCLUDED": (0, 0, 255),
                "CONTINUED_PATH": (255, 200, 0),
                "EXITED_FOV": (0, 255, 255),
                "MERGED_GROUP": (0, 165, 255),
                "STATIONARY": (180, 180, 180),
            }
            bar_y = y2 + 42
            bar_x = x1
            bar_w = min(x2 - x1, 120)
            bar_h = 8
            _safe_rect(out, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, (40, 40, 40))
            cx_off = 0
            for hyp in sorted(hyps, key=lambda h: h.get("probability", 0), reverse=True):
                prob = hyp.get("probability", 0)
                if not math.isfinite(prob) or prob <= 0:
                    continue
                seg_w = max(1, int(round(bar_w * prob)))
                seg_color = hyp_colors.get(hyp.get("type", ""), (200, 200, 200))
                _safe_rect(out, bar_x + cx_off, bar_y, bar_x + cx_off + seg_w, bar_y + bar_h, seg_color)
                cx_off += seg_w
            # Label with top hypothesis
            best_h = max(hyps, key=lambda h: h.get("probability", 0))
            hyp_label = f"{best_h.get('type', '?')} {best_h.get('probability', 0):.0%}"
            cv2.putText(out, hyp_label, (bar_x, bar_y + bar_h + 10), font, 0.25,
                        hyp_colors.get(best_h.get("type", ""), (200, 200, 200)), 1, cv2.LINE_AA)

        # Behavior label
        behavior_pattern = d.get("behavior_pattern", "")
        if behavior_pattern and behavior_pattern != "NORMAL_WALKING":
            pat_colors = {
                "LOITERING": (0, 255, 255),
                "EVASIVE": (0, 0, 255),
                "STOP_AND_GO": (0, 200, 200),
                "CONCEALMENT_SEEKING": (0, 100, 255),
                "ROAD_FOLLOWING": (200, 200, 0),
                "GROUP_MOVEMENT": (200, 100, 0),
                "ERRATIC": (0, 0, 255),
                "STATIONARY_OBSERVATION": (180, 180, 180),
            }
            pat_color = pat_colors.get(behavior_pattern, (200, 200, 200))
            tag_y = y2 + 56 if hyps else y2 + 42
            cv2.putText(out, behavior_pattern[:10], (x1, tag_y), font, 0.3, pat_color, 1, cv2.LINE_AA)

        # Priority score
        pri = d.get("priority_score", 0.0)
        if pri > 0.01:
            pri_y = y2 + 66 if hyps else y2 + 52
            pri_color = (0, 0, 255) if pri > 0.6 else (0, 200, 200)
            cv2.putText(out, f"PRI:{pri:.0%}", (x1, pri_y), font, 0.3, pri_color, 1, cv2.LINE_AA)

        # Concealment affinity
        conceal = d.get("concealment_affinity", 0.0)
        if conceal > 0.3:
            conc_y = y2 + 76 if hyps else y2 + 62
            cv2.putText(out, f"C:{conceal:.0%}", (x1 + 65, conc_y), font, 0.25, (0, 100, 255), 1, cv2.LINE_AA)

        # Anomaly indicator
        anomaly = d.get("anomaly_score", 0.0)
        if anomaly > 0.5:
            anom_y = y2 + 76 if hyps else y2 + 62
            cv2.putText(out, f"!{anomaly:.0%}", (x1 + 110, anom_y), font, 0.25, (0, 0, 255), 1, cv2.LINE_AA)

        # Merged-group indicator
        merge_group = d.get("merge_group", [])
        if merge_group:
            for other in detections:
                if other.get("track_id") in merge_group:
                    ox1, oy1, ox2, oy2 = [int(v) for v in other["bbox"]]
                    ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                    tcx, tcy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.line(out, (tcx, tcy), (ocx, ocy), (0, 165, 255), 1, cv2.LINE_AA)
                    mid_x, mid_y = (tcx + ocx) // 2, (tcy + ocy) // 2
                    cv2.putText(out, "MERGE", (mid_x - 15, mid_y - 5), font, 0.3, (0, 165, 255), 1, cv2.LINE_AA)

    # HUD bar
    cv2.rectangle(out, (0, 0), (w, 50), (0, 0, 0), cv2.FILLED)
    line1 = f"ISR  Cap:{capture_fps:.0f} Proc:{process_fps:.1f} Inf:{inference_ms:.0f}ms Dev:{device}"
    cv2.putText(out, line1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    line2 = f"Frm:{frame_id} Act:{active_track_count} Lost:{lost_track_count} Reaq:{reacquired_count} Uniq:{unique_track_count}"
    cv2.putText(out, line2, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)

    # Follow mode overlay
    if follow_id:
        if follow_target:
            fx1, fy1, fx2, fy2 = [int(v) for v in follow_target["bbox"]]
            fcx = (fx1 + fx2) // 2
            fcy = (fy1 + fy2) // 2

            # Tracking reticle
            cv2.circle(out, (fcx, fcy), 30, (255, 255, 255), 2)
            cv2.line(out, (fcx - 40, fcy), (fcx - 15, fcy), (255, 255, 255), 2)
            cv2.line(out, (fcx + 15, fcy), (fcx + 40, fcy), (255, 255, 255), 2)
            cv2.line(out, (fcx, fcy - 40), (fcx, fcy - 15), (255, 255, 255), 2)
            cv2.line(out, (fcx, fcy + 15), (fcx, fcy + 40), (255, 255, 255), 2)

            # Offset from center
            offset_x = fcx - w // 2
            offset_y = fcy - h // 2
            cv2.arrowedLine(
                out, (w // 2, h // 2), (fcx, fcy),
                (0, 255, 255), 1, tipLength=0.1,
            )

            # Lock indicator
            lock_text = f"LOCKED ON {follow_id}"
            (ltw, _), _ = cv2.getTextSize(lock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(
                out, lock_text,
                (w // 2 - ltw // 2, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
            offset_text = f"Offset: ({offset_x:+d}, {offset_y:+d})  Speed: {follow_target.get('speed', 0):.0f} px/f"
            cv2.putText(
                out, offset_text,
                (w // 2 - 150, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA,
            )

            # Predicted position circle if target has velocity
            vel = follow_target.get("velocity", [0, 0])
            spd = follow_target.get("speed", 0)
            if spd > 1.0:
                pred_scale = 15
                pred_x = int(fcx + vel[0] * pred_scale)
                pred_y = int(fcy + vel[1] * pred_scale)
                cv2.circle(out, (pred_x, pred_y), 8, (0, 165, 255), 2)
                cv2.putText(out, "PRED", (pred_x + 10, pred_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1, cv2.LINE_AA)

            # Standoff radius indicator + deterrence marker
            if follow_diag is not None and follow_diag.follow_mode.value == "standoff":
                bbox_h = follow_target["bbox"][3] - follow_target["bbox"][1]
                standoff_r = max(int(bbox_h * 0.8), 40)
                phase = follow_diag.standoff_phase
                ring_color = {
                    "APPROACHING": (0, 255, 255),
                    "SAFE": (0, 255, 0),
                    "ORBITING": (255, 255, 0),
                    "RETREAT": (0, 0, 255),
                    "HOLD": (100, 100, 100),
                }.get(phase, (150, 150, 150))
                # Dashed standoff radius circle
                num_pts = 60
                for i in range(num_pts):
                    if i % 3 == 0:
                        continue  # gap
                    a1 = 2 * math.pi * i / num_pts
                    a2 = 2 * math.pi * (i + 1) / num_pts
                    p1 = (int(fcx + standoff_r * math.cos(a1)), int(fcy + standoff_r * math.sin(a1)))
                    p2 = (int(fcx + standoff_r * math.cos(a2)), int(fcy + standoff_r * math.sin(a2)))
                    cv2.line(out, p1, p2, ring_color, 1, cv2.LINE_AA)
                # Phase label near standoff ring
                if phase:
                    cv2.putText(out, phase, (fcx + standoff_r + 5, fcy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ring_color, 1, cv2.LINE_AA)
                # Deterrence marker: pulsing concentric circles
                if follow_diag.deterrence_active:
                    pulse = int(8 + 6 * math.sin(frame_id * 0.2))
                    cv2.circle(out, (fcx, fcy), standoff_r + pulse, (0, 0, 255), 2)
                    cv2.putText(out, "DETER", (fcx - standoff_r - 30, fcy + standoff_r + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            # Target not found — blinking search indicator
            if frame_id % 10 < 5:
                search_text = f"SEARCHING FOR {follow_id}..."
                (stw, _), _ = cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(
                    out, search_text,
                    (w // 2 - stw // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )

    # Follow mode diagnostics
    if follow_diag is not None and follow_diag.state.value != "UNLOCKED":
        fd = follow_diag
        is_xy = fd.follow_mode.value in ("yaw_xy", "intercept")
        is_standoff = fd.follow_mode.value == "standoff"
        n_lines = 7 if (is_xy or is_standoff) else 5
        if is_standoff:
            n_lines = 8
        if hasattr(fd, "world_position") and fd.world_position:
            n_lines += 1
        if hasattr(fd, "hypotheses") and fd.hypotheses:
            n_lines += 1
        if hasattr(fd, "merge_group_partners") and fd.merge_group_partners:
            n_lines += 1
        if hasattr(fd, "priority_score") and fd.priority_score > 0:
            n_lines += 1
        bar_h = 18 * n_lines + 10
        cv2.rectangle(out, (0, h - bar_h), (w, h), (0, 0, 0), cv2.FILLED)
        state_color = follow_state_colors.get(fd.state.value, (200, 200, 200))
        y = h - bar_h + 18
        # Line 1: State + mode + offboard
        cv2.putText(out, f"MODE: {fd.state.value} ({fd.follow_mode.value})", (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)
        cv2.putText(out, f"OFFBOARD: {'ACTIVE' if fd.offboard_active else 'INACTIVE'}",
                    (320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0) if fd.offboard_active else (0, 0, 255), 1, cv2.LINE_AA)
        y += 18
        # Line 2: Yaw command + error
        yaw_deg_s = math.degrees(fd.yaw_rate)
        cv2.putText(out, f"YAW:{fd.yaw_rate:+.3f}rad/s ({yaw_deg_s:+.1f}d/s)  ERR:{fd.error_x:+.0f}px  SPD:{fd.target_speed:.0f}px/f",
                    (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
        y += 18
        # Line 3 (XY/intercept): velocity + bbox size
        if is_xy:
            fence_color = (0, 255, 0) if fd.geofence_ok else (0, 0, 255)
            intercept_str = ""
            if fd.follow_mode.value == "intercept" and fd.intercept_state:
                intercept_str = f"  INT:{fd.intercept_state}"
            standoff_str = ""
            if fd.follow_mode.value == "intercept":
                standoff_str = f"  STANDOFF:{'OK' if fd.standoff_distance_ok else 'FAR'}"
            cv2.putText(out, f"VX:{fd.vx:+.2f} VY:{fd.vy:+.2f} m/s  BBOX_H:{fd.bbox_height_px:.0f}px  SIZE_ERR:{fd.size_error:+.0f}px{intercept_str}",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(out, f"FENCE:{fd.distance_from_home_m:.0f}m {'OK' if fd.geofence_ok else 'BREACH'}{standoff_str}",
                        (w - 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, fence_color, 1, cv2.LINE_AA)
            y += 18
        # Standoff-specific lines
        if is_standoff:
            phase_color = {
                "APPROACHING": (0, 255, 255),
                "SAFE": (0, 255, 0),
                "ORBITING": (255, 255, 0),
                "RETREAT": (0, 0, 255),
                "HOLD": (150, 150, 150),
            }.get(fd.standoff_phase, (200, 200, 200))
            fence_color = (0, 255, 0) if fd.geofence_ok else (0, 0, 255)
            phase_str = fd.standoff_phase or "---"
            cv2.putText(out, f"PHASE: {phase_str}  VX:{fd.vx:+.2f} VY:{fd.vy:+.2f} VZ:{fd.vz:+.2f} m/s",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, phase_color, 1, cv2.LINE_AA)
            y += 18
            standoff_status = "AT DIST" if fd.standoff_distance_ok else "FAR"
            cv2.putText(out, f"STANDOFF: {fd.standoff_target_m:.0f}m tgt / alt {fd.standoff_altitude_m:.0f}m  BBOX_H:{fd.bbox_height_px:.0f}px  {standoff_status}",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(out, f"FENCE:{fd.distance_from_home_m:.0f}m {'OK' if fd.geofence_ok else 'BREACH'}",
                        (w - 180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, fence_color, 1, cv2.LINE_AA)
            y += 18
            det_str = "ON" if fd.deterrence_active else "OFF"
            cv2.putText(out, f"DETERRENCE: {det_str}",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 0) if fd.deterrence_active else (150, 150, 150), 1, cv2.LINE_AA)
            y += 18
        # Identity + similarity
        identity_text = f"ID: {fd.identity_descriptor or 'unknown'}  SIM: {fd.identity_similarity:.2f}"
        cv2.putText(out, identity_text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 0) if fd.identity_similarity > 0.6 else (150, 150, 150), 1, cv2.LINE_AA)
        y += 18
        # World-space target position
        if hasattr(fd, "world_position") and fd.world_position:
            wp = fd.world_position
            wv = getattr(fd, "world_velocity", None)
            occ = getattr(fd, "occlusion_state", "")
            world_text = f"WORLD: {wp.get('lat', 0):.5f},{wp.get('lon', 0):.5f} d:{wp.get('uncertainty_m', 0):.0f}m unc"
            if wv and wv.get("speed_mps", 0) > 0.1:
                world_text += f" v:{wv['speed_mps']:.1f}m/s"
            if occ and occ != "VISIBLE":
                world_text += f" [{occ}]"
            cv2.putText(out, world_text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (180, 180, 255), 1, cv2.LINE_AA)
            y += 18
        # Hypothesis breakdown for followed target
        if hasattr(fd, "hypotheses") and fd.hypotheses:
            hyp_parts = "  ".join(
                f"{h.get('type', '?')[:4]}:{h.get('probability', 0):.0%}"
                for h in sorted(fd.hypotheses, key=lambda h: h.get("probability", 0), reverse=True)[:4]
            )
            cv2.putText(out, f"HYP: {hyp_parts}", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (200, 200, 0), 1, cv2.LINE_AA)
            y += 18
        if hasattr(fd, "merge_group_partners") and fd.merge_group_partners:
            cv2.putText(out, f"MERGE: {','.join(fd.merge_group_partners)}", (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1, cv2.LINE_AA)
            y += 18
        # Behavioral diagnostics
        if hasattr(fd, "priority_score") and fd.priority_score > 0:
            pat_str = getattr(fd, "behavior_pattern", "")
            pri_str = f"PRI:{fd.priority_score:.0%}"
            anom_str = ""
            conc_str = ""
            # Find the followed target for behavioral data
            if follow_target:
                anom_str = f" ANOM:{follow_target.get('anomaly_score', 0):.0%}" if follow_target.get("anomaly_score", 0) > 0.5 else ""
                conc_str = f" CONC:{follow_target.get('concealment_affinity', 0):.0%}" if follow_target.get("concealment_affinity", 0) > 0.3 else ""
            beh_text = f"BEHAVIOR: {pat_str or '---'} {pri_str}{anom_str}{conc_str}"
            cv2.putText(out, beh_text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (200, 150, 0), 1, cv2.LINE_AA)
            y += 18
        # PID + search + confidence
        search_dir_label = "RIGHT" if fd.search_direction > 0 else "LEFT"
        lost_str = f"  LOST:{fd.target_lost_s:.1f}s" if fd.target_lost_s > 0 else ""
        cv2.putText(out, f"PID: I={fd.pid_state.get('integral', 0):.3f}  CONF:{fd.target_confidence:.2f}  SCAN:{search_dir_label}{lost_str}",
                    (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
        y += 18
        # Controls reminder
        cv2.putText(out, "[f]lock [a]aim [o]standoff [m]manual [TAB]cycle [s]stop [q]quit", (w - 610, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    return out


def _draw_isr_lite(
    frame: np.ndarray,
    detections: list[dict],
    capture_fps: float,
    process_fps: float,
    inference_ms: float,
    device: str,
    frame_id: int,
    unique_track_count: int = 0,
    active_track_count: int = 0,
    lost_track_count: int = 0,
    reacquired_count: int = 0,
    follow_id: str | None = None,
    timing: dict | None = None,
    debug_world_projection: bool = False,
) -> np.ndarray:
    """Lightweight ISR overlay — boxes, labels, HUD only. No trajectories/hypotheses/behavior."""
    out = frame.copy()
    h, w = out.shape[:2]

    status_colors = {
        "CONFIRMED": (0, 220, 0),
        "LOST": (0, 0, 255),
        "TENTATIVE": (0, 255, 255),
    }
    follow_state_colors = {
        "LOCKED": (0, 255, 0), "SEARCHING": (0, 255, 255),
        "LOST": (0, 0, 255), "REACQUIRED": (255, 255, 0),
    }

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        status = d.get("status", "TENTATIVE")
        tid = str(d.get("track_id", "?"))
        is_followed = follow_id is not None and tid == follow_id

        if follow_id and not is_followed:
            color = (60, 60, 60)
        else:
            color = status_colors.get(status, (0, 200, 200))

        thickness = 2 if is_followed else 1
        if status == "LOST":
            _draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness)
        else:
            _safe_rect(out, x1, y1, x2, y2, color, thickness)

        label = f"{tid} {d['class']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
        _safe_rect(out, x1, y1 - th - 6, x1 + tw + 4, y1, color)
        cv2.putText(out, label, (x1 + 2, y1 - 3), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        if debug_world_projection:
            proj_status = d.get("projection_status", "NEVER_ATTEMPTED")
            status_color = (0, 255, 0) if proj_status == "OK" else (0, 0, 255)
            cv2.putText(out, f"WORLD: {proj_status}", (x1, y2 + 14), font, 0.3, status_color, 1, cv2.LINE_AA)

    # HUD bar
    _safe_rect(out, 0, 0, w, 28, (0, 0, 0))
    line1 = f"ISR-LITE  Cap:{capture_fps:.0f} Proc:{process_fps:.1f} Inf:{inference_ms:.0f}ms Dev:{device} Frm:{frame_id} Act:{active_track_count} Lost:{lost_track_count}"
    cv2.putText(out, line1, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # Timing bar (bottom)
    if timing:
        bar_h = 18
        _safe_rect(out, 0, h - bar_h, w, h, (0, 0, 0))
        parts = [f"det:{timing.get('det_ms', 0):.0f}"]
        parts.append(f"trk:{timing.get('trk_ms', 0):.0f}")
        if "reid_ms" in timing:
            parts.append(f"rid:{timing['reid_ms']:.0f}")
        if "hyp_ms" in timing:
            parts.append(f"hyp:{timing['hyp_ms']:.0f}")
        if "beh_ms" in timing:
            parts.append(f"beh:{timing['beh_ms']:.0f}")
        parts.append(f"ovr:{timing.get('overlay_ms', 0):.0f}")
        parts.append(f"shw:{timing.get('imshow_ms', 0):.0f}")
        cv2.putText(out, "  ".join(parts), (6, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YOLO detection on Gazebo camera feed")
    parser.add_argument("--topic", required=True, help="Gazebo image topic")
    parser.add_argument("--backend", choices=["yolo", "mock"], default="yolo", help="Detection backend")
    parser.add_argument("--model-path", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--classes", nargs="+", default=["person", "car", "truck", "bus"],
                        help="COCO classes to detect")
    parser.add_argument("--max-frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for first frame")
    parser.add_argument("--save-annotated", metavar="DIR", help="Save annotated frames as JPEGs")
    parser.add_argument("--target-fps", type=float, default=0,
                        help="Max processing FPS (0=auto: 15 for CUDA, 5 for CPU)")
    parser.add_argument("--process-every", type=int, default=1,
                        help="Process every Nth captured frame (default: 1)")
    parser.add_argument("--track", choices=["bytetrack", "botsort", "iou", "persistent", "isr"], help="Enable tracking")
    parser.add_argument("--follow-track", metavar="ID", help="Lock camera on track ID (e.g., TGT-001) and show follow overlay")
    parser.add_argument("--follow-mode", choices=["overlay", "yaw", "yaw_xy", "intercept", "standoff", "manual_aim"], default="overlay",
                        help="Follow mode: overlay=visual only, yaw=aim lock, yaw_xy=aim+translation, intercept=safe approach+orbit, standoff=safe observation at distance, manual_aim=operator position + automatic aim (default: overlay)")
    parser.add_argument("--standoff-distance", type=float, default=10.0,
                        help="Standoff distance in meters (default: 10)")
    parser.add_argument("--standoff-altitude", type=float, default=10.0,
                        help="Standoff altitude above target in meters (default: 10)")
    parser.add_argument("--min-airborne-alt", type=float, default=3.0,
                        help="Minimum relative altitude (m) considered airborne for follow enable (default: 3.0)")
    parser.add_argument("--orbit-on-arrival", action="store_true", default=False,
                        help="In standoff mode, orbit once at standoff distance")
    parser.add_argument("--deterrence-marker", action="store_true", default=False,
                        help="Show deterrence marker circle on target when at standoff")
    parser.add_argument("--intercept-track", metavar="ID", help="Safe intercept: approach target and orbit at standoff distance")
    parser.add_argument("--auto-follow-nearest-person", action="store_true",
                        help="Automatically follow nearest person after tracker stabilizes")
    parser.add_argument("--mavlink-url", default="udpin://0.0.0.0:14540",
                        help="MAVSDK connection URL for offboard follow (default: udpin://0.0.0.0:14540)")
    parser.add_argument("--yaw-max-rate", type=float, default=0.6,
                        help="Max yaw rate in rad/s for follow mode (default: 0.6)")
    parser.add_argument("--follow-kp", type=float, default=0.008, help="PID proportional gain (default: 0.008)")
    parser.add_argument("--follow-ki", type=float, default=0.0001, help="PID integral gain (default: 0.0001)")
    parser.add_argument("--follow-kd", type=float, default=0.003, help="PID derivative gain (default: 0.003)")
    parser.add_argument("--follow-target-size", type=float, default=120.0,
                        help="Target bbox height in px at desired follow distance (default: 120)")
    parser.add_argument("--follow-max-vxy", type=float, default=0.8,
                        help="Max horizontal velocity m/s for yaw_xy mode (default: 0.8)")
    parser.add_argument("--follow-geofence-radius", type=float, default=50.0,
                        help="Max distance from home in meters (default: 50)")
    parser.add_argument("--follow-min-confidence", type=float, default=0.3,
                        help="Min track confidence for XY movement (default: 0.3)")
    parser.add_argument("--intercept-standoff-size", type=float, default=200.0,
                        help="Target bbox height px at standoff distance for intercept (default: 200)")
    parser.add_argument("--intercept-orbit-rate", type=float, default=0.15,
                        help="Orbit yaw rate rad/s at standoff (default: 0.15)")
    parser.add_argument("--track-buffer", type=int, default=None, help="Tracker track_buffer (frames)")
    parser.add_argument("--match-thresh", type=float, default=None, help="Tracker match_thresh")
    parser.add_argument("--track-high", type=float, default=None, help="Tracker track_high_thresh")
    parser.add_argument("--track-low", type=float, default=None, help="Tracker track_low_thresh")
    parser.add_argument("--max-lost-frames", type=int, default=90, help="Persistent tracker: frames to keep lost tracks (default: 90)")
    parser.add_argument("--min-hits", type=int, default=3, help="Persistent tracker: hits before ACTIVE (default: 3)")
    parser.add_argument("--reid", choices=["colorhist", "none"], default="colorhist", help="Persistent tracker: re-id method (default: colorhist)")
    parser.add_argument("--iou-thresh", type=float, default=0.2, help="Persistent tracker: IoU threshold (default: 0.2)")
    parser.add_argument("--center-dist-thresh", type=float, default=100.0, help="Persistent tracker: center distance threshold px (default: 100)")
    parser.add_argument("--no-display", action="store_true", help="Disable live overlay window")
    parser.add_argument("--lite-overlay", action="store_true",
                        help="Lightweight ISR overlay: boxes + labels + HUD only (no trajectories/hypotheses/behavior)")
    parser.add_argument("--overlay-every", type=int, default=1,
                        help="Draw overlay every Nth processed frame (default: 1)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="YOLO inference image size (e.g. 320, 416, 640). Default: model native")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-frame timing breakdown: det/trk/reid/hyp/beh/ovr/imshow")
    parser.add_argument("--follow-dry-run", action="store_true",
                        help="Compute follow diagnostics without calling MAVSDK/offboard (vision loop is never blocked)")
    parser.add_argument("--use-workers", action="store_true",
                        help="Use CameraWorker + PerceptionWorker for decoupled ingest/inference")
    parser.add_argument("--debug-target-selection", action="store_true",
                        help="Print target selection debug info every second")
    parser.add_argument("--debug-world-projection", action="store_true",
                        help="Show per-track projection status and expanded projection metrics")
    args = parser.parse_args()

    if not _GZ_AVAILABLE:
        print("ERROR: gz.transport13 not available", file=sys.stderr)
        sys.exit(1)
    if not _CV2_AVAILABLE:
        print("ERROR: opencv-python not available", file=sys.stderr)
        sys.exit(1)

    # Resolve device and target FPS
    device = _resolve_device(args.device)
    if args.target_fps <= 0:
        if args.track == "isr":
            args.target_fps = 15.0
        elif args.track == "iou":
            args.target_fps = 20.0
        elif device == "cuda":
            args.target_fps = 15.0
        else:
            args.target_fps = 5.0
    print(f"Device: {device}")
    print(f"Target FPS: {args.target_fps}")
    if args.lite_overlay:
        print("Overlay: lite mode (no trajectories/hypotheses/behavior)")
    if args.overlay_every > 1:
        print(f"Overlay: every {args.overlay_every} frames")
    if args.imgsz:
        print(f"YOLO imgsz: {args.imgsz}")

    # Validate backend
    yolo_model = None
    if args.backend == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics not installed. Use --backend mock or: pip install ultralytics",
                  file=sys.stderr)
            sys.exit(1)
        yolo_model = YOLO(args.model_path)
        print(f"Model: {args.model_path}")
        model_classes = getattr(yolo_model, "names", {})
        if model_classes:
            print(f"  Model classes: {dict(model_classes)}")
            aliases = {k: v for k, v in _CLASS_ALIASES.items() if k in model_classes.values()}
            if aliases:
                print(f"  Class aliases enabled: {aliases}")
            else:
                print(f"  Class aliases: none needed")

    # Setup IoU tracker if requested
    iou_tracker = None
    if args.track == "iou":
        from sentinel.tracker.simple_tracker import SimpleIoUTracker

        iou_tracker = SimpleIoUTracker(
            iou_threshold=0.3,
            max_lost_frames=30,
            min_confidence=args.confidence,
            prediction_enabled=True,
            distance_threshold_px=150,
            fps=args.target_fps,
            bbox_smoothing_enabled=True,
            bbox_ema_alpha=0.35,
            confidence_ema_alpha=0.35,
            fast_confirm_confidence=0.85,
            reacquire_window_frames=40,
            class_aware_matching=True,
        )
        print("IoU tracker: enabled (stable IDs, smoothed boxes)")

    # Setup persistent tracker if requested
    persistent_tracker = None
    if args.track == "persistent":
        from sentinel.tracker.persistent import PersistentTrackManager

        persistent_tracker = PersistentTrackManager(
            iou_thresh=args.iou_thresh,
            center_dist_thresh=args.center_dist_thresh,
            max_lost_frames=args.max_lost_frames,
            min_hits=args.min_hits,
            reid_method=args.reid,
        )
        print(f"Persistent tracker: enabled (reid={args.reid}, max_lost={args.max_lost_frames}, min_hits={args.min_hits})")

    # Setup ISR tracker if requested
    isr_tracker = None
    if args.track == "isr":
        from sentinel.tracker.isr_tracker import ISRTracker
        from sentinel.tracker.world_projection import CameraParams

        camera_params = CameraParams()
        isr_tracker = ISRTracker(
            max_lost_frames=args.max_lost_frames,
            min_hits_confirm=args.min_hits,
            fps=args.target_fps,
            camera=camera_params,
        )
        print(f"ISR tracker: enabled (max_lost={args.max_lost_frames}, min_hits={args.min_hits}, cmc=True)")

    # Geospatial layer for world-space trail storage
    geospatial = None
    if args.track == "isr":
        from sentinel.tracker.geospatial import GeospatialLayer
        geospatial = GeospatialLayer()

    # Setup follow controller for autonomous follow
    follow_controller = None
    follow_backend = None

    def _map_follow_mode(follow_mode: str):
        from sentinel.autonomy.follow_controller import FollowMode
        return {
            "intercept": FollowMode.INTERCEPT,
            "standoff": FollowMode.STANDOFF,
            "yaw_xy": FollowMode.YAW_XY,
            "manual_aim": FollowMode.MANUAL_AIM,
        }.get(follow_mode, FollowMode.YAW)

    operator_lock_mode = "yaw" if args.follow_mode == "overlay" else args.follow_mode

    def _current_mode_label() -> str:
        return operator_lock_mode

    # ── Worker follow path: deferred until after FlightBridgeWorker creation ──

    # ── Legacy dry-run follow path ─────────────────────────────────────
    if not args.use_workers and args.follow_dry_run and args.follow_mode in ("yaw", "yaw_xy", "intercept", "standoff", "manual_aim"):
        from sentinel.mavlink_bridge.mock_backend import MockMavlinkBackend
        from sentinel.autonomy.follow_controller import FollowController

        follow_backend = MockMavlinkBackend()
        follow_backend.connect()
        mode = _map_follow_mode(args.follow_mode)
        follow_controller = FollowController(
            backend=follow_backend,
            follow_mode=mode,
            kp=args.follow_kp, ki=args.follow_ki, kd=args.follow_kd,
            max_yaw_rate=args.yaw_max_rate,
            target_height_px=args.follow_target_size,
            max_vxy=args.follow_max_vxy,
            min_confidence=args.follow_min_confidence,
            geofence_radius_m=args.follow_geofence_radius,
            standoff_bbox_height_px=args.intercept_standoff_size,
            orbit_yaw_rate=args.intercept_orbit_rate,
            standoff_distance_m=args.standoff_distance,
            standoff_altitude_m=args.standoff_altitude,
            standoff_approach_speed=min(args.follow_max_vxy, 1.0),
            orbit_on_arrival=args.orbit_on_arrival,
            deterrence_marker=args.deterrence_marker,
        )
        print(f"Follow controller: DRY-RUN {mode.value} mode (mock backend, no MAVSDK)")
        print(f"  Runtime mode:    legacy follow path")
        print(f"  MAVSDK owner:    {type(follow_backend).__name__}")

    # ── Legacy MAVSDK follow path ──────────────────────────────────────
    elif not args.use_workers and args.follow_mode in ("yaw", "yaw_xy", "intercept", "standoff", "manual_aim") and args.track == "isr":
        _REQUIRED_OFFBOARD = ("offboard_start", "offboard_stop",
                              "offboard_set_velocity_body", "offboard_is_active")
        try:
            from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend
            from sentinel.autonomy.follow_controller import FollowController

            follow_backend = MavsdkBackend(connection_url=args.mavlink_url)

            # Startup self-test: verify backend has required offboard methods
            missing = [m for m in _REQUIRED_OFFBOARD if not hasattr(follow_backend, m)]
            if missing:
                print(f"FATAL: {type(follow_backend).__name__} missing offboard methods: {missing}")
                print(f"  Backend class: {type(follow_backend).__name__}")
                print(f"  Module: {inspect.getfile(type(follow_backend))}")
                print(f"  Required: {_REQUIRED_OFFBOARD}")
                print(f"  Available: {[m for m in dir(follow_backend) if 'offboard' in m]}")
                sys.exit(1)

            print(f"  Backend: {type(follow_backend).__name__} from {inspect.getfile(type(follow_backend))}")
            print(f"  Offboard self-test: PASS ({', '.join(_REQUIRED_OFFBOARD)})")

            connect_result = follow_backend.connect()
            if connect_result.success:
                mode = _map_follow_mode(args.follow_mode)
                follow_controller = FollowController(
                    backend=follow_backend,
                    follow_mode=mode,
                    kp=args.follow_kp, ki=args.follow_ki, kd=args.follow_kd,
                    max_yaw_rate=args.yaw_max_rate,
                    target_height_px=args.follow_target_size,
                    max_vxy=args.follow_max_vxy,
                    min_confidence=args.follow_min_confidence,
                    geofence_radius_m=args.follow_geofence_radius,
                    standoff_bbox_height_px=args.intercept_standoff_size,
                    orbit_yaw_rate=args.intercept_orbit_rate,
                    standoff_distance_m=args.standoff_distance,
                    standoff_altitude_m=args.standoff_altitude,
                    standoff_approach_speed=min(args.follow_max_vxy, 1.0),
                    orbit_on_arrival=args.orbit_on_arrival,
                    deterrence_marker=args.deterrence_marker,
                )
                print(f"Follow controller: MAVSDK connected, {mode.value} mode ready (kp={args.follow_kp})")
                print(f"  Runtime mode:    legacy follow path")
                print(f"  MAVSDK owner:    {type(follow_backend).__name__}")
            else:
                print(f"WARN: MAVSDK connection failed ({connect_result.message}), falling back to overlay mode")
                args.follow_mode = "overlay"
        except ImportError as e:
            print(f"WARN: MAVSDK not available ({e}), falling back to overlay mode")
            args.follow_mode = "overlay"

    # Handle --intercept-track shortcut
    if args.intercept_track:
        args.follow_track = args.intercept_track
        if args.follow_mode == "overlay":
            args.follow_mode = "intercept"

    # Setup output directory
    annotated_dir = None
    if args.save_annotated:
        annotated_dir = Path(args.save_annotated)
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # Subscribe to camera
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # Worker-based ingest + inference (Phase 2 decoupled path)
    camera_worker = None
    perception_worker = None
    flight_bridge = None
    state = None

    if args.use_workers:
        from sentinel.runtime.workers.camera_worker import CameraWorker
        from sentinel.runtime.workers.perception_worker import PerceptionWorker

        camera_worker = CameraWorker(topic=args.topic)
        print(f"Subscribing to: {args.topic} (CameraWorker)")
        if not camera_worker.connect(timeout_s=args.timeout):
            print(f"FAIL: no frame within {args.timeout}s", file=sys.stderr)
            sys.exit(2)

        perception_worker = PerceptionWorker(
            frame_slot=camera_worker.frame_slot,
            meta_slot=camera_worker.meta_slot,
            backend=args.backend,
            model_path=args.model_path,
            device=device,
            confidence=args.confidence,
            classes=args.classes,
            imgsz=args.imgsz,
            hz=args.target_fps,
        )
        if args.backend == "yolo":
            perception_worker.load_model()
        perception_worker.start()
        camera_worker.start()
        print(f"Workers started: camera + perception ({args.target_fps} Hz)")

        # FlightBridgeWorker — isolates MAVSDK from vision loop
        if args.follow_mode in ("yaw", "yaw_xy", "intercept", "standoff", "manual_aim"):
            from sentinel.runtime.workers.flight_bridge_worker import FlightBridgeWorker

            if args.follow_dry_run:
                flight_bridge = FlightBridgeWorker(
                    backend="mock", mock_connected=True, name="flight_bridge_mock",
                )
                print("FlightBridge: mock mode (dry-run)")
            else:
                flight_bridge = FlightBridgeWorker(url=args.mavlink_url)
                if flight_bridge.connect(timeout_s=10):
                    print(f"FlightBridge: connected to {args.mavlink_url}")
                else:
                    print("WARN: FlightBridge connect failed, falling back to overlay")
                    flight_bridge = None
                    args.follow_mode = "overlay"

            if flight_bridge is not None:
                flight_bridge.start()
                print(f"FlightBridge started (offboard {flight_bridge._hz:.0f} Hz)")

        # ── Worker follow controller (after FlightBridgeWorker) ────────
        if flight_bridge is not None and args.follow_mode in ("yaw", "yaw_xy", "intercept", "standoff", "manual_aim"):
            from sentinel.autonomy.follow_controller import FollowController

            mode = _map_follow_mode(args.follow_mode)
            follow_controller = FollowController(
                command_slot=flight_bridge.command_slot,
                telemetry_cache=flight_bridge.telemetry_cache,
                follow_mode=mode,
                kp=args.follow_kp, ki=args.follow_ki, kd=args.follow_kd,
                max_yaw_rate=args.yaw_max_rate,
                target_height_px=args.follow_target_size,
                max_vxy=args.follow_max_vxy,
                min_confidence=args.follow_min_confidence,
                geofence_radius_m=args.follow_geofence_radius,
                standoff_bbox_height_px=args.intercept_standoff_size,
                orbit_yaw_rate=args.intercept_orbit_rate,
                standoff_distance_m=args.standoff_distance,
                standoff_altitude_m=args.standoff_altitude,
                standoff_approach_speed=min(args.follow_max_vxy, 1.0),
                orbit_on_arrival=args.orbit_on_arrival,
                deterrence_marker=args.deterrence_marker,
            )
            print(f"Follow controller: worker {mode.value} mode (command-sink, no MAVSDK backend)")
            print(f"  Runtime mode:    workers")
            print(f"  Follow path:     command-slot")
            print(f"  MAVSDK owner:    FlightBridgeWorker")
    else:
        # Legacy inline path
        state = FrameState()
        print(f"Subscribing to: {args.topic}")
        node = _gzt.Node()
        node.subscribe(msg_type=_image_pb2.Image, topic=args.topic, callback=state.on_msg)

        print(f"Waiting up to {args.timeout}s for first frame...")
        if not state.connected.wait(timeout=args.timeout):
            print(f"FAIL: no frame within {args.timeout}s", file=sys.stderr)
            sys.exit(2)

    print(f"Connected. Processing {args.max_frames} frames with {args.backend} backend...")
    print(f"  Classes: {args.classes}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Class aliases: {_CLASS_ALIASES}")
    print(f"  Tracking: {args.track or 'disabled'}")
    if args.follow_track:
        print(f"  Follow target: {args.follow_track} (mode: {args.follow_mode})")
    if args.auto_follow_nearest_person:
        print(f"  Auto-follow: enabled (nearest person after 30 frames)")
    print()

    # Setup display window
    show_display = not args.no_display
    if show_display:
        window_name = "YOLO Gazebo ISR Test  [f]lock [a]aim [o]standoff [m]manual [TAB]cycle [s]stop [q]quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

    t_start = time.monotonic()
    processed = 0
    total_detections = 0
    class_counts: Counter = Counter()
    raw_class_counts: Counter = Counter()
    conf_values: list[float] = []
    active_track_ids: set[int | str] = set()
    first_person_frame = -1
    class_track_ids = defaultdict(set)
    class_confidences = defaultdict(list)
    frame_interval = 1.0 / args.target_fps
    last_process = 0.0
    last_recv_count = 0

    # FPS tracking
    capture_fps = 0.0
    process_fps = 0.0
    inference_ms = 0.0
    last_fps_time = t_start
    last_fps_process_count = 0

    # Auto-lock follow target if specified and follow controller is active
    # (Delay lock until we have a few frames for tracks to stabilize)
    follow_lock_pending = args.follow_track and follow_controller is not None
    if follow_lock_pending:
        print(f"  Follow lock pending: will lock {args.follow_track} after 10 frames")
    last_fps_recv_count = 0
    frame_skip_counter = 0
    last_key_f_time = 0.0
    last_key_tab_time = 0.0
    last_debug_target_time = 0.0

    # Profiling accumulators
    timing: dict[str, float] = {}
    timing_sum: dict[str, float] = {}
    timing_count: int = 0
    cached_overlay: np.ndarray | None = None
    _camera_params_updated = False

    # Build tracker config for ByteTrack/BotSort
    tracker_cfg_path: str | None = None
    if args.track in ("bytetrack", "botsort"):
        tracker_cfg_path = _build_tracker_config(args)

    while not stop.is_set() and processed < args.max_frames:
        now = time.monotonic()

        # Rate gate
        if now - last_process < frame_interval:
            time.sleep(0.001)
            continue

        # ── Frame + detection acquisition ──────────────────────────────
        if args.use_workers:
            # Worker path: read latest detection from PerceptionWorker
            det_val, det_seq, det_ts = perception_worker.detection_slot.read()
            if det_val is None:
                time.sleep(0.001)
                continue
            if det_seq == last_recv_count:
                time.sleep(0.001)
                continue

            last_process = now
            last_recv_count = det_seq

            # Get the corresponding frame
            frame, _, _ = camera_worker.frame_slot.read()
            if frame is None:
                continue

            recv_count = camera_worker.frame_slot.seq_id()

            detections = det_val.detections
            inference_ms = det_val.inference_ms
            timing["det_ms"] = det_val.inference_ms
            timing["trk_ms"] = 0.0

        else:
            # Legacy inline path
            frame, recv_count = state.snapshot()
            if frame is None:
                continue

            # Skip stale frames
            if recv_count == last_recv_count:
                time.sleep(0.001)
                continue

            # Process-every gate
            frame_skip_counter += 1
            if args.process_every > 1 and frame_skip_counter % args.process_every != 0:
                last_process = now
                last_recv_count = recv_count
                continue

            last_process = now
            last_recv_count = recv_count

        t0 = time.monotonic()

        # One-time camera params update from actual frame dimensions
        if not _camera_params_updated and isr_tracker is not None:
            _fh, _fw = frame.shape[:2]
            isr_tracker.update_camera_params(CameraParams(
                width_px=_fw, height_px=_fh,
                h_fov_deg=70.0, v_fov_deg=45.0, pitch_deg=-45.0,
            ))
            _camera_params_updated = True
            print(f"Camera params updated: {_fw}x{_fh}")

        # Run detection + tracking (with optional profiling)
        t_det_start = time.monotonic()
        if args.use_workers:
            # Workers already produced detections — only run tracker here
            if args.track == "isr" and isr_tracker is not None:
                t_trk_start = time.monotonic()
                # Read telemetry from FlightBridgeWorker cache (zero MAVSDK calls)
                if flight_bridge is not None:
                    tele_snap = flight_bridge.telemetry_cache.snapshot()
                    if tele_snap.position is not None and not tele_snap.stale(5.0):
                        isr_tracker.update_drone_pose(
                            lat=tele_snap.position.lat,
                            lon=tele_snap.position.lon,
                            alt_m=tele_snap.position.alt_m,
                            heading_deg=tele_snap.heading_deg or 0.0,
                            groundspeed_mps=tele_snap.groundspeed_mps or 0.0,
                        )
                detections = isr_tracker.update(detections, frame, processed)
                timing["trk_ms"] = (time.monotonic() - t_trk_start) * 1000
                if geospatial is not None:
                    geospatial.update(detections)
            elif args.track == "persistent" and persistent_tracker is not None:
                t_trk_start = time.monotonic()
                detections = persistent_tracker.update(detections, frame, processed)
                timing["trk_ms"] = (time.monotonic() - t_trk_start) * 1000
            elif args.track == "iou" and iou_tracker is not None:
                t_trk_start = time.monotonic()
                tracker_dets = _yolo_dets_to_tracker_dets(detections, processed)
                timestamp = datetime.now(timezone.utc).isoformat()
                iou_tracker.update(tracker_dets, processed, timestamp)
                active_tracks = iou_tracker.get_frame_active_tracks()
                detections = _tracker_tracks_to_dets(active_tracks)
                timing["trk_ms"] = (time.monotonic() - t_trk_start) * 1000

        elif args.backend == "yolo":
            if args.track == "isr":
                raw_dets, _ = _run_yolo_detection(
                    yolo_model, frame, args.confidence, args.classes, device,
                    imgsz=args.imgsz,
                )
                timing["det_ms"] = (time.monotonic() - t_det_start) * 1000

                t_trk_start = time.monotonic()
                if follow_backend is not None:
                    try:
                        tele = follow_backend.get_telemetry()
                        isr_tracker.update_drone_pose(
                            lat=tele.position.lat,
                            lon=tele.position.lon,
                            alt_m=tele.position.alt_m,
                            heading_deg=tele.heading_deg,
                            groundspeed_mps=tele.groundspeed_mps,
                        )
                    except Exception:
                        pass
                detections = isr_tracker.update(raw_dets, frame, processed)
                timing["trk_ms"] = (time.monotonic() - t_trk_start) * 1000

                if geospatial is not None:
                    geospatial.update(detections)
            elif args.track == "persistent":
                raw_dets, _ = _run_yolo_detection(
                    yolo_model, frame, args.confidence, args.classes, device,
                    imgsz=args.imgsz,
                )
                timing["det_ms"] = (time.monotonic() - t_det_start) * 1000

                t_trk_start = time.monotonic()
                detections = persistent_tracker.update(raw_dets, frame, processed)
                timing["trk_ms"] = (time.monotonic() - t_trk_start) * 1000
            elif args.track == "iou":
                detections, _ = _run_yolo_detection(
                    yolo_model, frame, args.confidence, args.classes, device,
                    imgsz=args.imgsz,
                )
                timing["det_ms"] = (time.monotonic() - t_det_start) * 1000

                t_trk_start = time.monotonic()
                tracker_dets = _yolo_dets_to_tracker_dets(detections, processed)
                timestamp = datetime.now(timezone.utc).isoformat()
                iou_tracker.update(tracker_dets, processed, timestamp)
                active_tracks = iou_tracker.get_frame_active_tracks()
                detections = _tracker_tracks_to_dets(active_tracks)
                timing["trk_ms"] = (time.monotonic() - t_trk_start) * 1000
            elif args.track in ("bytetrack", "botsort"):
                track_kwargs: dict = {
                    "conf": args.confidence, "classes": None,
                    "tracker": tracker_cfg_path, "persist": True,
                    "verbose": False, "device": device,
                }
                if args.imgsz:
                    track_kwargs["imgsz"] = args.imgsz
                results = yolo_model.track(frame, **track_kwargs)
                detections, _ = _extract_detections(results, args.classes)
                timing["det_ms"] = (time.monotonic() - t_det_start) * 1000
                timing["trk_ms"] = 0.0
            else:
                detections, _ = _run_yolo_detection(
                    yolo_model, frame, args.confidence, args.classes, device,
                    imgsz=args.imgsz,
                )
                timing["det_ms"] = (time.monotonic() - t_det_start) * 1000
        else:
            detections = _run_mock_backend(frame)
            timing["det_ms"] = (time.monotonic() - t_det_start) * 1000

        dt = time.monotonic() - t0
        inference_ms = dt * 1000
        processed += 1
        total_detections += len(detections)

        for d in detections:
            class_counts[d["class"]] += 1
            raw_class_counts[d.get("raw_class", d["class"])] += 1
            conf_values.append(d["confidence"])
            class_confidences[d["class"]].append(d["confidence"])
            if d.get("track_id") is not None:
                active_track_ids.add(d["track_id"])
                class_track_ids[d["class"]].add(d["track_id"])
            if d["class"] == "person" and first_person_frame == -1:
                first_person_frame = processed

        # Auto-lock follow target after tracker stabilizes
        if follow_lock_pending and processed >= 25:
            target_found = any(d.get("track_id") == args.follow_track for d in detections)
            if target_found:
                if flight_bridge:
                    _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                follow_controller.lock_target(args.follow_track)
                follow_lock_pending = False
                print(f"Follow: auto-locked on {args.follow_track}")

        # FPS calculation
        now_mono = time.monotonic()
        fps_dt = now_mono - last_fps_time
        if fps_dt >= 0.5:
            process_fps = (processed - last_fps_process_count) / fps_dt
            capture_fps = (recv_count - last_fps_recv_count) / fps_dt
            last_fps_time = now_mono
            last_fps_process_count = processed
            last_fps_recv_count = recv_count

        # Track counts for HUD
        current_active = sum(1 for d in detections if d.get("track_id") is not None and d.get("status") in ("ACTIVE", "active", "REACQUIRED", "CONFIRMED"))
        current_lost = sum(1 for d in detections if d.get("status") in ("LOST", "lost"))
        current_reacq = 0
        if persistent_tracker:
            m = persistent_tracker.get_metrics()
            current_reacq = m["total_reacquisitions"]
        if isr_tracker:
            m = isr_tracker.get_metrics()
            current_reacq = m["total_recoveries"]

        # Follow controller integration
        follow_diag = None
        if follow_controller is not None:
            fh, fw = frame.shape[:2]
            follow_controller.update_target(detections, fw, fh)
            follow_diag = follow_controller.get_diagnostics()

        # Auto-follow nearest person (priority-aware when behavioral data available)
        if args.auto_follow_nearest_person and args.follow_track is None:
            if processed >= 30 and detections:
                persons = [d for d in detections if d["class"] == "person" and d.get("track_id")]
                if persons:
                    # Use priority scoring if behavioral data is available
                    scored_persons = [(d, d.get("priority_score", 0.0)) for d in persons]
                    max_pri = max(s for _, s in scored_persons) if scored_persons else 0
                    if max_pri > 0.1:
                        best = max(scored_persons, key=lambda x: x[1])[0]
                    else:
                        best = min(persons, key=lambda d: abs(d["bbox"][0] + d["bbox"][2] - frame.shape[1]))
                    args.follow_track = best["track_id"]
                    if follow_controller:
                        follow_controller.lock_target(args.follow_track)
                    if flight_bridge:
                        _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                    print(f"Auto-follow: locked on {args.follow_track} (priority={best.get('priority_score', 0):.2f})")

        # Draw overlay
        t_ovr_start = time.monotonic()
        if processed % args.overlay_every == 0 or cached_overlay is None:
            if args.track == "isr" and args.lite_overlay:
                overlay = _draw_isr_lite(
                    frame, detections, capture_fps, process_fps, inference_ms,
                    device, processed,
                    unique_track_count=len(active_track_ids),
                    active_track_count=current_active,
                    lost_track_count=current_lost,
                    reacquired_count=current_reacq,
                    follow_id=args.follow_track,
                    timing=timing,
                    debug_world_projection=args.debug_world_projection,
                )
            elif args.track == "isr":
                overlay = _draw_isr_overlay(
                    frame, detections, capture_fps, process_fps, inference_ms,
                    device, processed,
                    unique_track_count=len(active_track_ids),
                    active_track_count=current_active,
                    lost_track_count=current_lost,
                    reacquired_count=current_reacq,
                    follow_id=args.follow_track,
                    follow_diag=follow_diag,
                    debug_world_projection=args.debug_world_projection,
                )
            else:
                overlay = _draw_overlay(
                    frame, detections, capture_fps, process_fps, inference_ms,
                    device, processed,
                    tracker_type=args.track,
                    unique_track_count=len(active_track_ids),
                    active_track_count=current_active,
                    lost_track_count=current_lost,
                    reacquired_count=current_reacq,
                )
            cached_overlay = overlay
        else:
            overlay = cached_overlay

        # Target selection HUD on overlay
        if args.follow_track or follow_controller:
            fh, fw = overlay.shape[:2]
            sel_count = len(_selectable_targets(detections, fw))
            follow_state = follow_controller.state.value if follow_controller else "N/A"
            lost_s = 0.0
            if follow_controller and follow_controller._lost_time > 0:
                lost_s = time.monotonic() - follow_controller._lost_time
            _draw_target_hud(
                overlay,
                selectable_count=sel_count,
                selected_target=args.follow_track,
                locked_target=args.follow_track if follow_controller and follow_controller.state.value != "UNLOCKED" else None,
                follow_on=follow_controller is not None and follow_controller.state.value != "UNLOCKED",
                autonomy_on=flight_bridge is not None and flight_bridge.autonomy_enabled,
                follow_state=follow_state,
                lost_time_s=lost_s,
            )

        # Telemetry HUD (always show when flight bridge exists)
        if flight_bridge is not None:
            _draw_telemetry_hud(overlay, flight_bridge)
            # Big warning banner if not airborne
            ok, reason = _check_airborne(flight_bridge, args.min_airborne_alt)
            if not ok:
                h, w = overlay.shape[:2]
                warn_text = f"NOT AIRBORNE: {reason}"
                sub_text = "Arm and Take Off in QGC first"
                (tw, th), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cx = w // 2
                cy = h // 2
                cv2.rectangle(overlay, (cx - tw // 2 - 10, cy - th - 10), (cx + tw // 2 + 10, cy + th + 10), (0, 0, 0), -1)
                cv2.putText(overlay, warn_text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(overlay, sub_text, (cx - tw2 // 2, cy + th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw LOCKED label on locked target
        if args.follow_track:
            for d in detections:
                if d.get("track_id") == args.follow_track:
                    bbox = d["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    # Red border for locked target
                    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 3)
                    cv2.putText(overlay, f"LOCKED {args.follow_track}", (x1, y1 - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    break

        timing["overlay_ms"] = (time.monotonic() - t_ovr_start) * 1000
        for k, v in timing.items():
            timing_sum[k] = timing_sum.get(k, 0.0) + v
        timing_count += 1

        # Save annotated frame
        if annotated_dir is not None and detections:
            cv2.imwrite(str(annotated_dir / f"frame_{processed:06d}.jpg"), overlay)

        # QGC / Manual flight mode override check (1 Hz)
        if processed % int(args.target_fps or 25) == 0 and flight_bridge is not None and flight_bridge.autonomy_enabled:
            tele_snap = flight_bridge.telemetry_cache.snapshot()
            if tele_snap.mode is not None:
                # If PX4 is not in OFFBOARD (e.g. manually switched to POSITION or stabilized/takeoff mode)
                if tele_snap.mode.upper() not in ("OFFBOARD", "UNKNOWN", "DISARMED", "HOLD", "TAKEOFF"):
                    print(f"\n[SAFETY] External manual stick / QGC pilot override detected (PX4 Mode: {tele_snap.mode})! Disengaging autonomy.")
                    flight_bridge.autonomy_enabled = False
                    if follow_controller:
                        follow_controller.unlock()
                    args.follow_track = None

        # Show display
        if show_display:
            t_imshow_start = time.monotonic()
            cv2.imshow(window_name, overlay)
            timing["imshow_ms"] = (time.monotonic() - t_imshow_start) * 1000
            key = cv2.pollKey()
            
            # Manual Piloting Keyboard Handler
            manual_pressed = False
            if follow_controller is not None:
                from sentinel.autonomy.follow_controller import FollowMode
                mvx, mvy, mvz = follow_controller._manual_vx, follow_controller._manual_vy, follow_controller._manual_vz
                
                # Detect Arrow keys (common OpenCV codes: Up=82/65362/38/2490368, Down=84/65364/40/2621440, etc.)
                if key in (82, 65362, 38, 2490368):
                    mvx = min(mvx + 0.15, 0.8)
                    manual_pressed = True
                elif key in (84, 65364, 40, 2621440):
                    mvx = max(mvx - 0.15, -0.8)
                    manual_pressed = True
                    
                if key in (81, 65361, 37, 2424832):
                    mvy = max(mvy - 0.15, -0.8)
                    manual_pressed = True
                elif key in (83, 65363, 39, 2555904):
                    mvy = min(mvy + 0.15, 0.8)
                    manual_pressed = True
                    
                if key == 91 or key == ord("["):  # Climb
                    mvz = min(mvz + 0.1, 0.5)
                    manual_pressed = True
                elif key == 93 or key == ord("]"):  # Descend
                    mvz = max(mvz - 0.1, -0.5)
                    manual_pressed = True
                    
                if manual_pressed:
                    if follow_controller.follow_mode != FollowMode.MANUAL_AIM:
                        follow_controller.set_follow_mode(FollowMode.MANUAL_AIM)
                        operator_lock_mode = "manual_aim"
                        print(f"Manual Pilot: switched FollowMode to MANUAL_AIM")
                        if flight_bridge:
                            _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                    follow_controller.set_manual_velocity(mvx, mvy, mvz)
                else:
                    # Decay manual velocities back to zero smoothly
                    decay = 0.82
                    mvx *= decay
                    mvy *= decay
                    mvz *= decay
                    if abs(mvx) < 0.05: mvx = 0.0
                    if abs(mvy) < 0.05: mvy = 0.0
                    if abs(mvz) < 0.05: mvz = 0.0
                    follow_controller.set_manual_velocity(mvx, mvy, mvz)

            if key == ord("q"):
                break
            elif key == ord("f"):
                now_key = time.monotonic()
                if now_key - last_key_f_time < 0.3:
                    pass  # debounce
                elif args.follow_track is None:
                    fh, fw = frame.shape[:2]
                    sel = _selectable_targets(detections, fw)
                    print(f"F pressed: selectable={len(sel)}")
                    if sel:
                        best = sel[0]
                        args.follow_track = best["track_id"]
                        if flight_bridge:
                            _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                        if follow_controller:
                            mode = _map_follow_mode(operator_lock_mode)
                            follow_controller.set_follow_mode(mode)
                            follow_controller.lock_target(args.follow_track)
                        last_key_f_time = now_key
                        print(f"Follow: locked on {args.follow_track} ({best['class']} conf={best['confidence']:.2f}) mode={follow_controller.follow_mode.value if follow_controller else _current_mode_label()}")
                    else:
                        last_key_f_time = now_key
                        print("Follow: no selectable targets")
                else:
                    if follow_controller:
                        follow_controller.unlock()
                    if flight_bridge:
                        flight_bridge.autonomy_enabled = False
                    args.follow_track = None
                    last_key_f_time = now_key
                    print("Follow: unlocked")
            elif key == ord("a"):
                # MODE 1 — Aim Lock / Camera Lock (Yaw only)
                if follow_controller:
                    from sentinel.autonomy.follow_controller import FollowMode
                    if args.follow_track is None:
                        fh, fw = frame.shape[:2]
                        sel = _selectable_targets(detections, fw)
                        if sel:
                            args.follow_track = sel[0]["track_id"]
                            operator_lock_mode = "yaw"
                            follow_controller.set_follow_mode(FollowMode.YAW)
                            follow_controller.lock_target(args.follow_track)
                            if flight_bridge:
                                _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                            print(f"Follow: Aim-Lock (YAW) ON for {args.follow_track}")
                    elif follow_controller.follow_mode == FollowMode.YAW:
                        follow_controller.unlock()
                        if flight_bridge:
                            flight_bridge.autonomy_enabled = False
                        args.follow_track = None
                        print("Follow: Aim-Lock OFF")
                    else:
                        operator_lock_mode = "yaw"
                        follow_controller.set_follow_mode(FollowMode.YAW)
                        print(f"Follow: switched to Aim-Lock (YAW) for {args.follow_track}")
            elif key == ord("o"):
                # MODE 2 — Standoff Follow
                if follow_controller:
                    from sentinel.autonomy.follow_controller import FollowMode
                    if args.follow_track is None:
                        fh, fw = frame.shape[:2]
                        sel = _selectable_targets(detections, fw)
                        if sel:
                            args.follow_track = sel[0]["track_id"]
                            operator_lock_mode = "standoff"
                            follow_controller.set_follow_mode(FollowMode.STANDOFF)
                            follow_controller.lock_target(args.follow_track)
                            if flight_bridge:
                                _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                            print(f"Follow: Standoff-Follow ON for {args.follow_track}")
                    elif follow_controller.follow_mode == FollowMode.STANDOFF:
                        follow_controller.unlock()
                        if flight_bridge:
                            flight_bridge.autonomy_enabled = False
                        args.follow_track = None
                        print("Follow: Standoff-Follow OFF")
                    else:
                        operator_lock_mode = "standoff"
                        follow_controller.set_follow_mode(FollowMode.STANDOFF)
                        print(f"Follow: switched to Standoff-Follow for {args.follow_track}")
            elif key == ord("m"):
                # MODE 3 — Manual Position + Aim Lock
                if follow_controller:
                    from sentinel.autonomy.follow_controller import FollowMode
                    if args.follow_track is None:
                        fh, fw = frame.shape[:2]
                        sel = _selectable_targets(detections, fw)
                        if sel:
                            args.follow_track = sel[0]["track_id"]
                            operator_lock_mode = "manual_aim"
                            follow_controller.set_follow_mode(FollowMode.MANUAL_AIM)
                            follow_controller.lock_target(args.follow_track)
                            if flight_bridge:
                                _enable_autonomy_safe(flight_bridge, args.min_airborne_alt)
                            print(f"Follow: Manual-Aim (Mode 3) ON for {args.follow_track}")
                    elif follow_controller.follow_mode == FollowMode.MANUAL_AIM:
                        follow_controller.unlock()
                        if flight_bridge:
                            flight_bridge.autonomy_enabled = False
                        args.follow_track = None
                        print("Follow: Manual-Aim OFF")
                    else:
                        operator_lock_mode = "manual_aim"
                        follow_controller.set_follow_mode(FollowMode.MANUAL_AIM)
                        print(f"Follow: switched to Manual-Aim (Mode 3) for {args.follow_track}")
            elif key == ord("\t"):
                now_key = time.monotonic()
                if now_key - last_key_tab_time < 0.3:
                    pass  # debounce
                else:
                    last_key_tab_time = now_key
                    fh, fw = frame.shape[:2]
                    sel = _selectable_targets(detections, fw)
                    print(f"Tab pressed: selectable={len(sel)}")
                    if sel:
                        tids = [t["track_id"] for t in sel]
                        old_id = args.follow_track
                        if args.follow_track in tids:
                            idx = (tids.index(args.follow_track) + 1) % len(tids)
                        else:
                            idx = 0
                        args.follow_track = tids[idx]
                        if follow_controller:
                            follow_controller.unlock()
                            follow_controller.set_follow_mode(_map_follow_mode(operator_lock_mode))
                            follow_controller.lock_target(args.follow_track)
                        print(f"Follow: {old_id} → {args.follow_track}")
                    else:
                        print("Tab: no selectable targets")
            elif key == ord("s"):
                # Emergency stop
                if follow_controller:
                    follow_controller.emergency_stop()
                    args.follow_track = None
                    print("Follow: EMERGENCY STOP")

        # Console report
        elapsed = now_mono - t_start
        if processed % 10 == 0 or processed == args.max_frames:
            det_str = ", ".join(f"{k}:{v}" for k, v in class_counts.most_common())
            track_str = f"  tracks: {len(active_track_ids)}" if active_track_ids else ""
            profile_str = ""
            if args.profile:
                parts = [f"det:{timing.get('det_ms', 0):.1f}"]
                parts.append(f"trk:{timing.get('trk_ms', 0):.1f}")
                if "reid_ms" in timing:
                    parts.append(f"rid:{timing['reid_ms']:.1f}")
                if "hyp_ms" in timing:
                    parts.append(f"hyp:{timing['hyp_ms']:.1f}")
                if "beh_ms" in timing:
                    parts.append(f"beh:{timing['beh_ms']:.1f}")
                parts.append(f"ovr:{timing.get('overlay_ms', 0):.1f}")
                parts.append(f"shw:{timing.get('imshow_ms', 0):.1f}")
                profile_str = f"  [{', '.join(parts)}]"
            flight_str = ""
            if flight_bridge is not None:
                tele = flight_bridge.telemetry_cache.snapshot()
                alt = tele.position.alt_m if tele.position else None
                flight_str = f"  ARM:{int(tele.armed)} ALT:{alt:.1f}m MODE:{tele.mode}"
            print(
                f"  [{processed:4d}/{args.max_frames}] "
                f"recv: {recv_count}  "
                f"det: {len(detections)}  "
                f"total: {total_detections}  "
                f"inference: {inference_ms:.0f}ms  "
                f"cap: {capture_fps:.0f}  proc: {process_fps:.1f}  "
                f"({det_str}){track_str}{profile_str}{flight_str}"
            )

        # Debug target selection (1 Hz)
        if args.debug_target_selection and now_mono - last_debug_target_time >= 1.0:
            last_debug_target_time = now_mono
            fh, fw = frame.shape[:2]
            sel = _selectable_targets(detections, fw)
            vis_tids = [d.get("track_id") for d in detections if d.get("track_id") and d.get("status") not in ("LOST",)]
            follow_state = follow_controller.state.value if follow_controller else "N/A"
            lost_s = 0.0
            if follow_controller and follow_controller._lost_time > 0:
                lost_s = now_mono - follow_controller._lost_time
            cmd_age = flight_bridge.command_slot.age_s() if flight_bridge else float("inf")
            tracker_age = time.monotonic() - follow_controller._tracker_time if follow_controller and follow_controller._tracker_time > 0 else float("inf")
            print(
                f"  [TARGETS] visible={vis_tids} "
                f"selectable={len(sel)} "
                f"selected={args.follow_track} "
                f"locked={args.follow_track if follow_controller and follow_controller.state.value != 'UNLOCKED' else None} "
                f"follow={follow_state} "
                f"lost={lost_s:.1f}s "
                f"tracker_age={tracker_age:.2f}s "
                f"cmd_age={cmd_age:.2f}s"
            )

    if show_display:
        cv2.destroyAllWindows()

    # Cleanup workers
    if perception_worker:
        perception_worker.stop()
    if camera_worker:
        camera_worker.stop()
    if flight_bridge:
        flight_bridge.stop()

    # Cleanup follow controller
    if follow_controller:
        follow_controller.emergency_stop()
    if follow_backend:
        follow_backend.close()

    # Cleanup temp tracker config
    if tracker_cfg_path and tracker_cfg_path.startswith(tempfile.gettempdir()):
        try:
            Path(tracker_cfg_path).unlink()
        except OSError:
            pass

    # Summary
    elapsed = time.monotonic() - t_start
    if not args.use_workers:
        _, recv_count = state.snapshot()

    print()
    print("=" * 60)
    print("YOLO + Gazebo Test Summary")
    print("=" * 60)
    print(f"  Device:           {device}")
    print(f"  Model:            {args.model_path}")
    print(f"  Tracking:         {args.track or 'disabled'}")
    print(f"  Frames processed: {processed}")
    print(f"  Frames received:  {recv_count}")
    print(f"  Duration:         {elapsed:.1f}s")
    if elapsed > 0:
        print(f"  Avg process FPS:  {processed / elapsed:.1f}")
    if recv_count > 0 and elapsed > 0:
        print(f"  Avg capture FPS:  {recv_count / elapsed:.1f}")
    print(f"  Avg inference:    {sum(conf_values) and inference_ms:.1f}ms")
    print(f"  Total detections: {total_detections}")
    if active_track_ids:
        print(f"  Unique tracks:    {len(active_track_ids)}")
    if persistent_tracker:
        m = persistent_tracker.get_metrics()
        print(f"  Persistent metrics:")
        print(f"    Active:   {m['active']}")
        print(f"    Lost:     {m['lost']}")
        print(f"    Tentative: {m['tentative']}")
        print(f"    Reacquired total: {m['total_reacquisitions']}")
        print(f"    Unique confirmed: {m['unique_confirmed']}")
    if isr_tracker:
        m = isr_tracker.get_metrics()
        print(f"  ISR tracker metrics:")
        print(f"    Active:   {m['active']}")
        print(f"    Lost:     {m['lost']}")
        print(f"    Tentative: {m['tentative']}")
        print(f"    Recoveries total: {m['total_recoveries']}")
        print(f"    Total tracks created: {m['total_tracks_created']}")
        print(f"    Alive tracks: {m['alive_tracks']}")
        reid = m.get("reid", {})
        if reid:
            print(f"  Re-ID metrics:")
            print(f"    ID switches:           {reid.get('id_switches', 0)}")
            print(f"    Reacquisitions:        {reid.get('reacquisitions', 0)}")
            print(f"    Cross-lifecycle matches: {reid.get('cross_lifecycle_matches', 0)}")
            print(f"    Total identities:      {reid.get('total_identities', 0)}")
            print(f"    Identity memory:       {m.get('identity_memory_size', 0)} archived")
        wt = m.get("world_tracking", {})
        if wt:
            print(f"  World tracking:")
            print(f"    Tracks with world pos: {wt.get('tracks_with_world_pos', 0)}")
            print(f"    Occluded tracks:       {wt.get('occluded_tracks', 0)}")
            print(f"    Drone altitude:        {wt.get('drone_alt_m', 0):.1f}m")
            if args.debug_world_projection:
                print(f"    Projection successes:  {wt.get('projection_successes', 0)}")
                print(f"    Projection failures:   {wt.get('projection_failures', 0)}")
                fail_reasons = wt.get('projection_fail_reasons', {})
                if fail_reasons:
                    print(f"    Failure reasons:")
                    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                        print(f"      {reason}: {count}")
        hyp_m = m.get("hyp", {})
        if hyp_m:
            print(f"  Hypothesis metrics:")
            print(f"    Occlusion recovery rate: {hyp_m.get('occlusion_recovery_rate', 0):.1%}")
            print(f"    Merge recovery rate:     {hyp_m.get('merge_recovery_rate', 0):.1%}")
            print(f"    False reacquisition rate:{hyp_m.get('false_reacquisition_rate', 0):.1%}")
            print(f"    Total resolved:          {hyp_m.get('total_resolved', 0)}")
            print(f"    Avg hypothesis count:    {hyp_m.get('avg_hypothesis_count', 0):.1f}")
        beh_m = m.get("behavior", {})
        if beh_m:
            print(f"  Behavior metrics:")
            print(f"    Avg anomaly score:       {beh_m.get('avg_anomaly_score', 0):.3f}")
            print(f"    Total interactions:      {beh_m.get('total_interactions', 0)}")
            print(f"    Concealment events:      {beh_m.get('concealment_events', 0)}")
            print(f"    Avg target lifetime:     {beh_m.get('target_avg_lifetime_frames', 0):.0f} frames")
            patterns = beh_m.get("pattern_counts", {})
            if patterns:
                print(f"    Pattern distribution:")
                for pat, cnt in sorted(patterns.items(), key=lambda x: -x[1]):
                    print(f"      {pat}: {cnt}")
    if geospatial:
        print(f"  Geospatial layer:")
        print(f"    Targets tracked:      {geospatial.target_count}")
        print(f"    Total trail points:   {geospatial.total_trail_points}")
    print()
    print("  Detection counts by class (normalized):")
    for cls, count in class_counts.most_common():
        tracks_count = len(class_track_ids[cls])
        avg_conf = sum(class_confidences[cls]) / len(class_confidences[cls]) if class_confidences[cls] else 0.0
        print(f"    {cls}: detections={count}, tracks={tracks_count}, avg_conf={avg_conf:.3f}")
    if raw_class_counts:
        print("  Raw class counts:")
        for cls, count in raw_class_counts.most_common():
            print(f"    {cls}: {count}")
    if first_person_frame != -1:
        print(f"  First person detection frame index: {first_person_frame}")
    else:
        print("  First person detection frame index: N/A (no person detected)")
    print()
    if conf_values:
        print("  Confidence stats:")
        print(f"    min: {min(conf_values):.3f}")
        print(f"    max: {max(conf_values):.3f}")
        print(f"    avg: {sum(conf_values) / len(conf_values):.3f}")
    if annotated_dir:
        print(f"  Annotated frames: {annotated_dir}")
    if timing_count > 0 and args.profile:
        print()
        print("  Profile averages (over %d frames):" % timing_count)
        for k in ("det_ms", "trk_ms", "reid_ms", "hyp_ms", "beh_ms", "overlay_ms", "imshow_ms"):
            if k in timing_sum:
                avg = timing_sum[k] / timing_count
                print(f"    {k:12s}: {avg:6.1f} ms")

    # Pass/fail
    if args.backend == "yolo":
        person_detected = class_counts.get("person", 0) > 0
        car_detected = class_counts.get("car", 0) > 0
        print()
        if person_detected or car_detected:
            print("PASS: detections found")
        else:
            print("WARN: no person or car detections")


if __name__ == "__main__":
    main()
