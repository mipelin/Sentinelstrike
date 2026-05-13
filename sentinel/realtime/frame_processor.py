"""Frame processor — wraps perception backend for single-frame detection."""

from __future__ import annotations

from sentinel.common.time import utc_now_iso
from sentinel.common.types import Detection


class FrameProcessor:
    """Thin wrapper around a PerceptionBackend for single-frame processing."""

    def __init__(self, backend: object) -> None:
        self._backend = backend

    def process(self, frame, frame_id: int, timestamp_utc: str | None = None) -> list[Detection]:
        ts = timestamp_utc or utc_now_iso()
        return self._backend.detect_frame(frame, frame_id, ts)

    def close(self) -> None:
        self._backend.close()
