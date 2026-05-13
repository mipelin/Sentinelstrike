"""Managed video source — unified capture with health, reconnect, timeout, and timestamp tracking."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterator

import cv2
import numpy as np
from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.time import utc_now_iso
from sentinel.common.types import SensorFrame

from .types import SensorHealth, VideoSourceType

_FPS_WINDOW = 30


class ManagedVideoSource:
    """Unified video source supporting file, webcam, and RTSP with reconnect."""

    def __init__(
        self,
        source_type: VideoSourceType,
        source_label: str,
        *,
        file_path: str = "",
        webcam_index: int = 0,
        rtsp_url: str = "",
        frame_width: int | None = None,
        frame_height: int | None = None,
        reconnect_enabled: bool = True,
        reconnect_interval_s: float = 5.0,
        event_bus: EventBus | None = None,
        mission_id: str = "",
        target_fps: float = 5.0,
        read_timeout_ms: float = 5000.0,
    ) -> None:
        self._source_type = source_type
        self._source_label = source_label
        self._file_path = file_path
        self._webcam_index = webcam_index
        self._rtsp_url = rtsp_url
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._reconnect_enabled = reconnect_enabled
        self._reconnect_interval_s = reconnect_interval_s
        self._event_bus = event_bus
        self._mission_id = mission_id
        self._target_fps = target_fps
        self._read_timeout_ms = read_timeout_ms

        self._cap: cv2.VideoCapture | None = None
        self._health = SensorHealth(
            source_type=source_type,
            source_label=source_label,
        )
        self._frame_timestamps: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._last_frame_monotonic: float = 0.0
        self._frame_id = 0
        self._closed = False
        self._latest_frame_age_ms: float = 0.0

    def open(self) -> bool:
        if self._source_type == VideoSourceType.FILE:
            return self._open_file()
        if self._source_type == VideoSourceType.WEBCAM:
            return self._open_webcam()
        if self._source_type == VideoSourceType.RTSP:
            return self._open_rtsp()
        return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None

        t0 = time.monotonic()
        ret, frame = self._cap.read()
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if ret and frame is not None:
            self._latest_frame_age_ms = elapsed_ms
            self._on_frame(frame)
            return True, frame

        if self._source_type in (VideoSourceType.FILE,):
            return False, None

        if self._source_type in (VideoSourceType.WEBCAM, VideoSourceType.RTSP):
            self._health.dropped_frames += 1
            if self._reconnect_enabled:
                self._reconnect()
                if self._cap is not None:
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._latest_frame_age_ms = 0.0
                        self._on_frame(frame)
                        return True, frame
            return False, None

        return False, None

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._health.connected = False

    @property
    def health(self) -> SensorHealth:
        now = time.monotonic()
        stale_threshold = 2.0 / max(self._target_fps, 0.1) if self._target_fps > 0 else 60.0
        if self._last_frame_monotonic > 0:
            self._health.stale = (now - self._last_frame_monotonic) > stale_threshold
        self._health.latest_frame_age_ms = self._latest_frame_age_ms
        return self._health

    @property
    def latest_frame_age_ms(self) -> float:
        if self._last_frame_monotonic > 0:
            return (time.monotonic() - self._last_frame_monotonic) * 1000.0
        return 0.0

    def build_sensor_frame(self, frame: np.ndarray, capture_timestamp_utc: str | None = None) -> SensorFrame:
        return SensorFrame(
            frame_id=self._frame_id,
            capture_timestamp_monotonic=self._last_frame_monotonic,
            capture_timestamp_utc=capture_timestamp_utc or utc_now_iso(),
            receive_timestamp_monotonic=time.monotonic(),
            source=self._source_label,
            metadata={"source_type": self._source_type.value, "width": frame.shape[1], "height": frame.shape[0]},
        )

    def write_health_json(self, path) -> None:
        h = self.health
        data = {
            "connected": h.connected,
            "fps_estimate": round(h.fps_estimate, 2),
            "dropped_frames": h.dropped_frames,
            "reconnect_count": h.reconnect_count,
            "last_frame_utc": h.last_frame_utc,
            "stale": h.stale,
            "latest_frame_age_ms": round(h.latest_frame_age_ms, 2),
            "source_type": h.source_type.value,
            "source_label": h.source_label,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        ret, frame = self.read_frame()
        if not ret:
            raise StopIteration
        return {
            "frame_id": self._frame_id - 1,
            "frame": frame,
        }

    def _open_file(self) -> bool:
        try:
            cap = cv2.VideoCapture(self._file_path)
            if not cap.isOpened():
                logger.error("Cannot open video file: {}", self._file_path)
                return False
            self._cap = cap
            self._health.connected = True
            self._publish_event("sensor_connected", {"source": self._file_path})
            return True
        except Exception as e:
            logger.error("Failed to open video file: {} — {}", self._file_path, e)
            return False

    def _open_webcam(self) -> bool:
        try:
            cap = cv2.VideoCapture(self._webcam_index)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._read_timeout_ms)
            if not cap.isOpened():
                logger.warning("Webcam index {} not available", self._webcam_index)
                return False
            if self._frame_width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            if self._frame_height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self._cap = cap
            self._health.connected = True
            self._publish_event("sensor_connected", {"source": f"webcam:{self._webcam_index}"})
            return True
        except Exception as e:
            logger.warning("Failed to open webcam {}: {}", self._webcam_index, e)
            return False

    def _open_rtsp(self) -> bool:
        try:
            cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._read_timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, max(self._read_timeout_ms, 3000))
            if not cap.isOpened():
                logger.error("Cannot open RTSP stream: {}", self._rtsp_url)
                return False
            if self._frame_width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            if self._frame_height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self._cap = cap
            self._health.connected = True
            self._publish_event("sensor_connected", {"source": self._rtsp_url})
            return True
        except Exception as e:
            logger.error("Failed to open RTSP stream: {} — {}", self._rtsp_url, e)
            return False

    def _on_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        self._frame_timestamps.append(now)
        self._last_frame_monotonic = now
        self._frame_id += 1

        if len(self._frame_timestamps) >= 2:
            span = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if span > 0:
                self._health.fps_estimate = len(self._frame_timestamps) / span

        self._health.last_frame_utc = utc_now_iso()

    def _reconnect(self) -> None:
        self._health.connected = False
        self._publish_event("sensor_disconnected", {"source": self._source_label})
        logger.warning(
            "Sensor disconnected: {}, reconnecting in {:.1f}s",
            self._source_label, self._reconnect_interval_s,
        )

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        time.sleep(self._reconnect_interval_s)

        success = self.open()
        if success:
            self._health.reconnect_count += 1
            self._publish_event("sensor_reconnected", {"source": self._source_label})
            logger.info("Sensor reconnected: {}", self._source_label)

    def _publish_event(self, event_type: str, payload: dict) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            make_event(
                mission_id=self._mission_id,
                source="sensor",
                event_type=event_type,
                payload=payload,
            )
        )
