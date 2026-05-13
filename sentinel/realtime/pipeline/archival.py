"""Archival stage — frame saving, JSONL writing, disk budget."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sentinel.common.types import Detection, GeoObservation, Track, VehicleState
from sentinel.perception.drawing import draw_tactical_overlay


class ArchivalStage:
    """Handles frame saving and JSONL fallback writing."""

    def __init__(
        self,
        run_dir: Path,
        *,
        save_frames: bool = True,
        save_latest_frame: bool = True,
        save_annotated_every_n_frames: int = 1,
        disk_budget_mb: int = 2048,
    ) -> None:
        self._run_dir = run_dir
        self._save_frames = save_frames
        self._save_latest_frame = save_latest_frame
        self._save_every_n = max(1, save_annotated_every_n_frames)
        self._disk_budget_mb = disk_budget_mb
        self._frames_dir = run_dir / "frames"
        self._latest_frame_path = run_dir / "latest.jpg"

    def ensure_dirs(self) -> None:
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def save_annotated_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_utc: str,
        detections: list[Detection],
        tracks: list[Track],
        vehicle_state: VehicleState | None,
    ) -> np.ndarray:
        annotated = draw_tactical_overlay(
            frame,
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
            detections=detections,
            tracks=tracks,
            vehicle_state=vehicle_state,
        )
        if self._save_frames and (frame_id % self._save_every_n == 0):
            cv2.imwrite(str(self._frames_dir / f"frame_{frame_id:06d}.jpg"), annotated)
        if self._save_latest_frame:
            cv2.imwrite(str(self._latest_frame_path), annotated)
        return annotated

    def write_detections_fallback(self, detections: list[Detection], file) -> None:
        for det in detections:
            file.write(det.model_dump_json() + "\n")

    def write_tracks_fallback(self, tracks: list[Track], file) -> None:
        for trk in tracks:
            file.write(trk.model_dump_json() + "\n")

    def write_observations_fallback(self, observations: list[GeoObservation], file) -> None:
        for obs in observations:
            file.write(obs.model_dump_json() + "\n")
