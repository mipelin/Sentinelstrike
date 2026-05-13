"""Drawing helpers for annotated ISR-style video output."""

from __future__ import annotations

import math

import cv2
import numpy as np

from sentinel.common.types import Detection, Track, VehicleState

_DETECTION_COLOR = (84, 195, 255)
_ACTIVE_COLOR = (91, 214, 108)
_TENTATIVE_COLOR = (180, 180, 80)
_LOST_COLOR = (51, 170, 255)
_CONFIRMED_COLOR = (80, 220, 180)
_REJECTED_COLOR = (96, 96, 160)
_STALE_COLOR = (0, 140, 255)
_TEXT_PRIMARY = (225, 235, 235)
_TEXT_DARK = (16, 18, 20)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        _draw_bbox(
            out,
            (det.bbox_xyxy.x1, det.bbox_xyxy.y1, det.bbox_xyxy.x2, det.bbox_xyxy.y2),
            _DETECTION_COLOR,
            f"DET {det.class_name.upper()} {det.confidence:.2f}",
            thickness=1,
        )
    return out


def draw_tactical_overlay(
    frame: np.ndarray,
    *,
    frame_id: int,
    timestamp_utc: str,
    detections: list[Detection],
    tracks: list[Track],
    vehicle_state: VehicleState | None = None,
    selected_track_id: str | None = None,
) -> np.ndarray:
    out = frame.copy()

    # Telemetry header band
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (12, 18, 22), -1)
    header = _build_header(frame_id, timestamp_utc, vehicle_state)
    cv2.putText(out, header, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT_PRIMARY, 1, cv2.LINE_AA)

    for det in detections:
        _draw_bbox(
            out,
            (det.bbox_xyxy.x1, det.bbox_xyxy.y1, det.bbox_xyxy.x2, det.bbox_xyxy.y2),
            _DETECTION_COLOR,
            f"DET {det.class_name.upper()} {det.confidence:.2f}",
            thickness=1,
        )

    for trk in tracks:
        bbox = _track_bbox(trk)
        if bbox is None:
            continue
        is_tentative = trk.metadata.get("confirmation_state") == "TENTATIVE"
        lifecycle = _track_lifecycle(trk)
        color = _track_color(lifecycle, is_tentative)
        tag = "[TENT] " if is_tentative else ""
        label = f"{tag}{trk.track_id} {trk.class_name.upper()} {trk.confidence:.2f} {lifecycle}"
        thickness = 3 if trk.track_id == selected_track_id else 2
        if is_tentative:
            _draw_dashed_bbox(out, bbox, color, label, thickness=thickness)
        else:
            _draw_bbox(out, bbox, color, label, thickness=thickness)
        _draw_trail(out, trk.metadata.get("history_px", []), color, highlight=trk.track_id == selected_track_id)

    return out


def _build_header(frame_id: int, timestamp_utc: str, vehicle_state: VehicleState | None) -> str:
    if vehicle_state is None:
        return f"FRAME {frame_id:06d}  {timestamp_utc}"

    pos = vehicle_state.position
    pos_text = "--"
    if pos is not None:
        pos_text = f"{pos.lat:.5f},{pos.lon:.5f}"
    battery_text = "--"
    if vehicle_state.battery_pct is not None:
        battery_text = f"{vehicle_state.battery_pct:.0f}%"
    alt_text = "--"
    if pos is not None and pos.alt_m is not None:
        alt_text = f"{pos.alt_m:.0f}m"
    speed_text = "--"
    if vehicle_state.groundspeed_mps is not None:
        speed_text = f"{vehicle_state.groundspeed_mps:.1f}m/s"
    hdg_text = "--"
    if vehicle_state.heading_deg is not None:
        hdg_text = f"{vehicle_state.heading_deg:.0f}deg"
    return (
        f"FRAME {frame_id:06d}  {timestamp_utc}  "
        f"UAV {vehicle_state.vehicle_id}  {vehicle_state.mode}  "
        f"ALT {alt_text}  SPD {speed_text}  HDG {hdg_text}  BATT {battery_text}  POS {pos_text}"
    )


def _track_bbox(track: Track) -> tuple[int, int, int, int] | None:
    predicted = track.metadata.get("predicted_bbox")
    if track.status == "lost" and isinstance(predicted, dict):
        return (
            int(predicted.get("x1", 0)),
            int(predicted.get("y1", 0)),
            int(predicted.get("x2", 0)),
            int(predicted.get("y2", 0)),
        )
    if track.bbox_xyxy is not None:
        return (track.bbox_xyxy.x1, track.bbox_xyxy.y1, track.bbox_xyxy.x2, track.bbox_xyxy.y2)
    if isinstance(predicted, dict):
        return (
            int(predicted.get("x1", 0)),
            int(predicted.get("y1", 0)),
            int(predicted.get("x2", 0)),
            int(predicted.get("y2", 0)),
        )
    return None


def _track_lifecycle(track: Track) -> str:
    transition = str(track.metadata.get("last_transition", "")).upper()
    if track.status == "lost" and track.lost_frames >= 3:
        return "STALE"
    if transition == "REACQUIRED":
        return "REACQUIRED"
    if track.status == "lost":
        return "LOST"
    return "ACTIVE"


def _track_color(lifecycle: str, is_tentative: bool = False) -> tuple[int, int, int]:
    if is_tentative:
        return _TENTATIVE_COLOR
    if lifecycle == "REACQUIRED":
        return _CONFIRMED_COLOR
    if lifecycle == "LOST":
        return _LOST_COLOR
    if lifecycle == "STALE":
        return _STALE_COLOR
    if lifecycle == "REJECTED":
        return _REJECTED_COLOR
    return _ACTIVE_COLOR


def _draw_trail(
    frame: np.ndarray,
    points: list[list[float]] | list[tuple[float, float]],
    color: tuple[int, int, int],
    *,
    highlight: bool = False,
) -> None:
    if len(points) < 2:
        return
    pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
    cv2.polylines(frame, [pts], False, color, 2 if highlight else 1, cv2.LINE_AA)


def _draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    *,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    top = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _TEXT_DARK, 1, cv2.LINE_AA)


def _draw_dashed_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    *,
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    x1, y1, x2, y2 = bbox
    for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        _draw_dashed_line(frame, start, end, color, thickness, dash_len, gap_len)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    top = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _TEXT_DARK, 1, cv2.LINE_AA)


def _draw_dashed_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = max(int(math.sqrt(dx * dx + dy * dy)), 1)
    step = dash_len + gap_len
    for i in range(0, dist, step):
        sx = int(pt1[0] + dx * i / dist)
        sy = int(pt1[1] + dy * i / dist)
        end_i = min(i + dash_len, dist)
        ex = int(pt1[0] + dx * end_i / dist)
        ey = int(pt1[1] + dy * end_i / dist)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
