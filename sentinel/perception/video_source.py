"""Video source — reads frames from file or webcam via OpenCV."""

from __future__ import annotations

from collections.abc import Iterator

import cv2

from sentinel.common.time import utc_now_iso


class VideoSource:
    def __init__(
        self,
        source: str,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self._source = source
        self._frame_stride = max(frame_stride, 1)
        self._max_frames = max_frames
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        raw_source: str | int = self._source
        if isinstance(raw_source, str) and raw_source.isdigit():
            raw_source = int(raw_source)
        cap = cv2.VideoCapture(raw_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._source}")
        return cap

    def __iter__(self) -> Iterator[dict]:
        self._cap = self._open()
        frame_id = 0
        yielded = 0
        try:
            while True:
                if self._max_frames is not None and yielded >= self._max_frames:
                    break
                ret, frame = self._cap.read()
                if not ret:
                    break
                if frame_id % self._frame_stride != 0:
                    frame_id += 1
                    continue
                h, w = frame.shape[:2]
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                yield {
                    "frame_id": frame_id,
                    "timestamp_utc": utc_now_iso(),
                    "frame": frame,
                    "source_fps": fps if fps > 0 else None,
                    "width": w,
                    "height": h,
                }
                yielded += 1
                frame_id += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
