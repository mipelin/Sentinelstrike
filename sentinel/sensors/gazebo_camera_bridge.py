"""Gazebo camera bridge — subscribes to gz transport image topics and exposes frames via the ManagedVideoSource interface."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import make_event
from sentinel.common.time import utc_now_iso

from .types import SensorHealth, VideoSourceType

_FPS_WINDOW = 30

# Gazebo pixel_format_type enum values
_RGB_INT8 = 3
_RGBA_INT8 = 4
_BGRA_INT8 = 5
_RGB_INT16 = 6
_RGB_INT32 = 7
_BGR_INT8 = 8
_RGB8 = 9
_RGB16 = 10
_L_INT8 = 29

_FORMAT_NAMES = {
    _RGB_INT8: "RGB_INT8",
    _RGBA_INT8: "RGBA_INT8",
    _BGRA_INT8: "BGRA_INT8",
    _RGB_INT16: "RGB_INT16",
    _RGB_INT32: "RGB_INT32",
    _BGR_INT8: "BGR_INT8",
    _RGB8: "R8G8B8",
    _RGB16: "RGB16",
    _L_INT8: "L_INT8",
}

try:
    import gz.transport13 as _gzt
    import gz.msgs10.image_pb2 as _image_pb2

    _GZ_AVAILABLE = True
except ImportError:
    _GZ_AVAILABLE = False


def gz_transport_available() -> bool:
    return _GZ_AVAILABLE


class GazeboCameraBridge:
    """Subscribes to a Gazebo camera image topic and exposes frames via read_frame().

    Implements the same duck-typed interface as ManagedVideoSource so the
    realtime loop and factory can use it interchangeably.
    """

    def __init__(
        self,
        topic: str,
        *,
        event_bus: EventBus | None = None,
        mission_id: str = "",
        target_fps: float = 10.0,
        startup_timeout_s: float = 5.0,
        frame_timeout_s: float = 1.0,
    ) -> None:
        if not _GZ_AVAILABLE:
            raise RuntimeError(
                "gz.transport13 / gz.msgs10 not available — install python3-gz-transport13 python3-gz-msgs10"
            )
        self._topic = topic
        self._event_bus = event_bus
        self._mission_id = mission_id
        self._target_fps = target_fps
        self._startup_timeout_s = startup_timeout_s
        self._frame_timeout_s = frame_timeout_s

        self._lock = threading.Lock()
        self._frame_available = threading.Condition(self._lock)
        self._latest_frame: np.ndarray | None = None
        self._frame_id = 0
        self._closed = False
        self._first_frame_received = False
        self._metadata_logged = False

        self._frame_timestamps: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._last_frame_monotonic: float = 0.0
        self._latest_frame_age_ms: float = 0.0
        self._dropped_frames = 0

        self._health = SensorHealth(
            source_type=VideoSourceType.GAZEBO_CAMERA,
            source_label=topic,
        )

        self._node: object | None = None

    # --- lifecycle ---

    def open(self) -> bool:
        try:
            self._node = _gzt.Node()
            self._node.subscribe(
                msg_type=_image_pb2.Image,
                topic=self._topic,
                callback=self._on_image,
            )
            self._health.connected = True
            self._publish_event("sensor_connected", {"source": self._topic})
            logger.info("GazeboCameraBridge subscribed to {}", self._topic)
            return True
        except Exception as exc:
            logger.error("Failed to subscribe to {}: {}", self._topic, exc)
            self._health.connected = False
            return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        with self._frame_available:
            if self._closed:
                return False, None

            # Wait for first frame with startup timeout
            if not self._first_frame_received:
                deadline = time.monotonic() + self._startup_timeout_s
                while self._latest_frame is None and not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning(
                            "GazeboCameraBridge: no frame after {:.1f}s startup timeout (topic={})",
                            self._startup_timeout_s, self._topic,
                        )
                        return False, None
                    self._frame_available.wait(timeout=remaining)

                if self._closed:
                    return False, None
                if self._latest_frame is None:
                    return False, None

            # Subsequent frames: wait with frame timeout
            if self._latest_frame is None:
                deadline = time.monotonic() + self._frame_timeout_s
                while self._latest_frame is None and not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # No frame available but not EOF — caller should retry
                        return False, None
                    self._frame_available.wait(timeout=remaining)

                if self._closed:
                    return False, None
                if self._latest_frame is None:
                    return False, None

            frame = self._latest_frame
            self._latest_frame = None
            self._frame_id += 1

        now = time.monotonic()
        self._frame_timestamps.append(now)
        self._last_frame_monotonic = now
        self._latest_frame_age_ms = 0.0

        if len(self._frame_timestamps) >= 2:
            span = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if span > 0:
                self._health.fps_estimate = len(self._frame_timestamps) / span
        self._health.last_frame_utc = utc_now_iso()
        return True, frame

    def close(self) -> None:
        with self._frame_available:
            self._closed = True
            self._frame_available.notify_all()
        self._health.connected = False
        self._node = None
        self._publish_event("sensor_disconnected", {"source": self._topic})
        logger.info("GazeboCameraBridge closed (frames_read={}, dropped={})", self._frame_id, self._dropped_frames)

    # --- health interface (same as ManagedVideoSource) ---

    @property
    def health(self) -> SensorHealth:
        now = time.monotonic()
        stale_threshold = 2.0 / max(self._target_fps, 0.1) if self._target_fps > 0 else 60.0
        if self._last_frame_monotonic > 0:
            self._health.stale = (now - self._last_frame_monotonic) > stale_threshold
        self._health.latest_frame_age_ms = self._latest_frame_age_ms
        self._health.dropped_frames = self._dropped_frames
        return self._health

    @property
    def latest_frame_age_ms(self) -> float:
        if self._last_frame_monotonic > 0:
            return (time.monotonic() - self._last_frame_monotonic) * 1000.0
        return 0.0

    def write_health_json(self, path: Path) -> None:
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

    # --- gz transport callback ---

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

    def _on_image(self, msg: object) -> None:
        try:
            w = msg.width
            h = msg.height
            step = msg.step
            pixel_format = msg.pixel_format_type
            raw = bytes(msg.data)

            if w == 0 or h == 0 or len(raw) == 0:
                return

            if not self._metadata_logged:
                fmt_name = _FORMAT_NAMES.get(pixel_format, f"UNKNOWN({pixel_format})")
                logger.info(
                    "GazeboCameraBridge first image: {}x{} fmt={} step={} data_len={}",
                    w, h, fmt_name, step, len(raw),
                )
                self._metadata_logged = True

            channels = step // w if w > 0 else 3
            dtype = np.uint8
            if pixel_format in (_RGB_INT16, _RGB8):
                dtype = np.uint16
                channels = step // (w * 2) if w > 0 else 3
            elif pixel_format in (_RGB_INT32, _RGB16):
                dtype = np.uint32
                channels = step // (w * 4) if w > 0 else 3

            # Greyscale: expand to BGR for downstream compatibility
            if pixel_format == _L_INT8:
                grey = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
                frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.frombuffer(raw, dtype=dtype).reshape((h, w, channels))

                if pixel_format == _RGB_INT8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif pixel_format == _BGRA_INT8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif pixel_format == _RGBA_INT8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif pixel_format == _BGR_INT8:
                    pass  # Already BGR

            with self._frame_available:
                # Drop previous unread frame
                if self._latest_frame is not None:
                    self._dropped_frames += 1
                self._latest_frame = frame.copy()
                self._first_frame_received = True
                self._frame_available.notify_all()

        except Exception as exc:
            logger.debug("GazeboCameraBridge image decode error: {}", exc)
