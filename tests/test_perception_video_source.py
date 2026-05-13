"""Tests for video source."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.perception.video_source import VideoSource


def _create_video(path: Path, frames: int = 20, w=320, h=240) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (w, h))
    for _ in range(frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()
    return path


def test_opens_video_file(tmp_path):
    vid = _create_video(tmp_path / "test.mp4", frames=10)
    vs = VideoSource(str(vid))
    frames = list(vs)
    assert len(frames) == 10


def test_respects_max_frames(tmp_path):
    vid = _create_video(tmp_path / "test.mp4", frames=30)
    vs = VideoSource(str(vid), max_frames=5)
    frames = list(vs)
    assert len(frames) == 5


def test_respects_frame_stride(tmp_path):
    vid = _create_video(tmp_path / "test.mp4", frames=10)
    vs = VideoSource(str(vid), frame_stride=2)
    frames = list(vs)
    assert len(frames) == 5


def test_invalid_source_raises():
    vs = VideoSource("/nonexistent/path.mp4")
    with pytest.raises(RuntimeError, match="Cannot open"):
        list(vs)
