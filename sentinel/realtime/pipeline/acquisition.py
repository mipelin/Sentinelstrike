"""Acquisition stage — opens and reads from video source."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class VideoSource(Protocol):
    def open(self) -> bool: ...
    def read_frame(self) -> tuple[bool, np.ndarray | None]: ...
    def close(self) -> None: ...


class FileAcquisition:
    """Reads frames from a file path via OpenCV."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._source)
        return self._cap.isOpened()

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ProviderAcquisition:
    """Wraps an existing video source provider (ManagedVideoSource etc.)."""

    def __init__(self, provider: object) -> None:
        self._provider = provider

    def open(self) -> bool:
        return self._provider.open()  # type: ignore[union-attr]

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        return self._provider.read_frame()  # type: ignore[union-attr]

    def close(self) -> None:
        self._provider.close()  # type: ignore[union-attr]

    @property
    def health(self):
        return self._provider.health  # type: ignore[union-attr]

    def write_health_json(self, path) -> None:
        self._provider.write_health_json(path)  # type: ignore[union-attr]
