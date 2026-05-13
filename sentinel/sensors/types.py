"""Sensor data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VideoSourceType(StrEnum):
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"


@dataclass
class SensorHealth:
    connected: bool = False
    fps_estimate: float = 0.0
    dropped_frames: int = 0
    reconnect_count: int = 0
    last_frame_utc: str | None = None
    stale: bool = False
    source_type: VideoSourceType = VideoSourceType.FILE
    source_label: str = ""
    latest_frame_age_ms: float = 0.0
