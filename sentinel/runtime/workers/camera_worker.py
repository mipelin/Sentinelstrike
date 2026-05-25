"""CameraWorker — independent frame ingest from Gazebo camera topic.

Owns the Gazebo subscriber, converts protobuf messages to BGR numpy arrays,
and publishes both the raw frame and a FrameSnapshot metadata contract into
LatestSlots. Runs at source FPS — never blocks on inference/tracking/render.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from loguru import logger

from sentinel.runtime.blackboard import LatestSlot
from sentinel.runtime.contracts import FrameSnapshot
from sentinel.runtime.worker import Worker


def _msg_to_bgr(msg: object) -> np.ndarray | None:
    """Convert a Gazebo Image protobuf message to a BGR numpy array."""
    w = msg.width
    h = msg.height
    pixel_format = getattr(msg, "pixel_format_type", 0)
    raw = bytes(msg.data)

    if w == 0 or h == 0 or len(raw) == 0:
        return None

    _FORMAT_CHANNELS: dict[int, tuple[str, int, type]] = {
        3: ("RGB_INT8", 3, np.uint8),
        4: ("RGBA_INT8", 4, np.uint8),
        5: ("BGRA_INT8", 4, np.uint8),
        6: ("RGB_INT16", 3, np.uint16),
        7: ("RGB_INT32", 3, np.uint32),
        8: ("BGR_INT8", 3, np.uint8),
        9: ("R8G8B8", 3, np.uint8),
        10: ("RGB16", 3, np.uint16),
        29: ("L_INT8", 1, np.uint8),
    }

    try:
        import cv2
    except ImportError:
        return None

    info = _FORMAT_CHANNELS.get(pixel_format)
    if info is None:
        return None

    _, channels, dtype = info

    if pixel_format == 29:  # L_INT8
        grey = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    frame = np.frombuffer(raw, dtype=dtype).reshape((h, w, channels))

    if pixel_format in (3, 9):  # RGB
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if pixel_format == 4:  # RGBA
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if pixel_format == 5:  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


class CameraWorker(Worker):
    """Frame ingest worker.

    Subscribes to a Gazebo image topic and publishes every received frame
    into two LatestSlots:
      - ``frame_slot``: raw BGR numpy array (caller must copy if needed)
      - ``meta_slot``: FrameSnapshot metadata contract

    Usage::

        cam = CameraWorker(topic="/world/.../image")
        cam.connect(timeout_s=15)
        cam.start()
        ...
        frame, seq, ts = cam.frame_slot.read()
        meta = cam.meta_slot.read()
        cam.stop()
    """

    def __init__(
        self,
        topic: str,
        name: str = "camera_ingest",
        hz: float = 0,
    ) -> None:
        """Initialise CameraWorker.

        Args:
            topic: Gazebo image topic path.
            name: Worker name for diagnostics.
            hz: Target Hz. Pass 0 to run at source rate (no sleep between ticks).
                With hz=0 the tick() blocks on the Gazebo subscriber callback,
                so the loop rate equals the camera publish rate.
        """
        super().__init__(name=name, hz=hz)
        self._topic = topic
        self._node: Any = None
        self._frame_seq = 0
        self._connected = threading.Event()

        # Public LatestSlots — consumers read from these
        self.frame_slot: LatestSlot[np.ndarray] = LatestSlot()
        self.meta_slot: LatestSlot[FrameSnapshot] = LatestSlot()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, timeout_s: float = 15.0) -> bool:
        """Create Gazebo subscriber. Blocks until first frame or timeout."""
        try:
            import gz.transport13 as _gzt
            import gz.msgs10.image_pb2 as _image_pb2
        except ImportError:
            logger.error("gz.transport13 not available")
            return False

        self._node = _gzt.Node()
        self._node.subscribe(
            msg_type=_image_pb2.Image,
            topic=self._topic,
            callback=self._on_msg,
        )
        logger.info("CameraWorker subscribed to {}", self._topic)

        if self._connected.wait(timeout=timeout_s):
            logger.info("CameraWorker connected — receiving frames")
            return True
        logger.error("CameraWorker: no frame within {}s", timeout_s)
        return False

    def _on_msg(self, msg: object) -> None:
        """Gazebo subscriber callback — runs on the GZ transport thread."""
        bgr = _msg_to_bgr(msg)
        if bgr is None:
            return

        self._frame_seq += 1
        now_mono = time.monotonic()
        now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Publish raw frame
        self.frame_slot.write(bgr)

        # Publish metadata contract
        meta = FrameSnapshot(
            frame_id=self._frame_seq,
            seq=self._frame_seq,
            ts_monotonic=now_mono,
            ts_utc=now_utc,
            shape=bgr.shape,
        )
        self.meta_slot.write(meta)

        self._connected.set()

    def tick(self) -> None:
        """Worker tick — at source rate, frames arrive via _on_msg callback.

        With hz=0 (source rate), the Gazebo transport thread drives ingest.
        This tick is a no-op — we just need the worker thread alive so
        metrics and health are tracked.
        """
        # Frames arrive on the Gazebo transport thread via _on_msg().
        # The worker thread exists for health/metrics reporting.
        # Sleep briefly to avoid busy-waiting.
        self._stop_event.wait(timeout=0.1)

    def stop(self) -> None:
        """Stop the worker and release Gazebo subscriber."""
        super().stop()
        self._node = None
