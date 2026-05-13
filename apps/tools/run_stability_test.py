"""Long-run stability test — runs the real-time loop for a set duration and reports metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger


def _create_video_source(source: str, fps: float, width: int = 640, height: int = 480):
    """Create a looping video source for sustained testing."""
    import cv2
    import numpy as np

    class LoopVideoSource:
        def __init__(self, src: str, target_fps: float):
            self._src = src
            self._target_fps = target_fps
            self._cap: cv2.VideoCapture | None = None
            self._frame_interval = 1.0 / max(target_fps, 0.1)
            self._last_read: float = 0.0
            self._dropped = 0
            self._frame_count = 0

        def open(self) -> bool:
            self._cap = cv2.VideoCapture(self._src)
            return self._cap.isOpened()

        def read_frame(self):
            now = time.monotonic()
            if now - self._last_read < self._frame_interval * 0.8:
                return True, None
            if self._cap is None:
                return False, None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Loop back
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            self._last_read = now
            if frame is not None:
                self._frame_count += 1
            return ret, frame

        def close(self) -> None:
            if self._cap is not None:
                self._cap.release()

        @property
        def health(self):
            class Health:
                dropped_frames = 0
                fps_estimate = self._target_fps
                reconnect_count = 0
            return Health()

        def write_health_json(self, path) -> None:
            pass

    class SyntheticVideoSource:
        """Generates synthetic frames when no video file available."""

        def __init__(self, target_fps: float, w: int, h: int):
            self._fps = target_fps
            self._w = w
            self._h = h
            self._frame_interval = 1.0 / max(target_fps, 0.1)
            self._last_read: float = 0.0
            self._frame_count = 0

        def open(self) -> bool:
            return True

        def read_frame(self):
            now = time.monotonic()
            if now - self._last_read < self._frame_interval * 0.8:
                return True, None
            self._last_read = now
            frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
            cv2.putText(frame, f"F{self._frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            self._frame_count += 1
            return True, frame

        def close(self) -> None:
            pass

        @property
        def health(self):
            class Health:
                dropped_frames = 0
                fps_estimate = self._fps
                reconnect_count = 0
            return Health()

        def write_health_json(self, path) -> None:
            pass

    import cv2
    import numpy as np

    if source and Path(source).exists():
        return LoopVideoSource(source, fps)
    return SyntheticVideoSource(fps, width, height)


def run_stability_test(
    duration_min: float,
    backend: str = "mock",
    video_source: str = "",
    target_fps: float = 10.0,
    save_frames: bool = True,
    report_every_s: float = 30.0,
    config_path: str | None = None,
) -> dict:
    """Run the real-time loop for a set duration and collect stability metrics."""
    from sentinel.common.event_bus import EventBus
    from sentinel.config.schema import AppConfig, PerceptionConfig, TrackerConfig
    from sentinel.perception.backends import PerceptionBackend
    from sentinel.realtime.loop import RealTimeMissionLoop
    from sentinel.tracker.runner import TrackingRunner

    config = AppConfig()
    if config_path:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        config = AppConfig(**data)

    run_dir = Path("runs") / f"stability_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    event_bus = EventBus()

    # Create backend
    if backend == "mock":
        from sentinel.perception.mock_backend import MockPerceptionBackend
        perc_backend = MockPerceptionBackend()
    elif backend == "yolo":
        from sentinel.perception.yolo_backend import YoloPerceptionBackend
        perc_backend = YoloPerceptionBackend(device=config.perception.device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    tracker_runner = TrackingRunner(
        config=config.tracker,
        mission_id="stability_test",
        event_bus=event_bus,
        run_dir=run_dir,
    )

    video_src = _create_video_source(video_source, target_fps)

    max_frames = int(duration_min * 60 * target_fps)

    loop = RealTimeMissionLoop(
        config=config,
        mission_id="stability_test",
        event_bus=event_bus,
        run_dir=run_dir,
        perception_backend=perc_backend,
        tracker_runner=tracker_runner,
        max_frames=max_frames,
        target_fps=target_fps,
        video_source=video_source or "synthetic",
        video_source_provider=video_src,
        save_frames=save_frames,
        save_latest_frame=save_frames,
    )

    start = time.monotonic()
    report_interval = report_every_s
    last_report = start

    logger.info(
        "Stability test: {} min, backend={}, fps={}, max_frames={}",
        duration_min, backend, target_fps, max_frames,
    )

    # Patch loop to report progress
    original_tick = loop._tick

    frame_count_ref = [0]
    tick_times: list[float] = []

    def tick_with_progress(**kwargs):
        result = original_tick(**kwargs)
        frame_count_ref[0] += 1
        tick_times.append(result.get("tick_ms", 0))
        now = time.monotonic()
        if now - last_report >= report_interval:
            elapsed = now - start
            fps = frame_count_ref[0] / max(elapsed, 0.001)
            logger.info(
                "Progress: {}/{} frames, {:.1f} fps, elapsed {:.1f}s",
                frame_count_ref[0], max_frames, fps, elapsed,
            )
        return result

    loop._tick = tick_with_progress

    # Run the loop
    metrics = loop.run()

    wall_s = time.monotonic() - start

    # Collect additional metrics
    ram_mb = 0.0
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    all_tracks = tracker_runner.get_all_tracks()
    unique_tracks = len({t.track_id for t in all_tracks})
    frame_count_actual = metrics.get("frame_count", frame_count_ref[0])
    id_churn = unique_tracks / max(frame_count_actual, 1)

    avg_tick = sum(tick_times) / len(tick_times) if tick_times else 0
    max_tick = max(tick_times) if tick_times else 0
    min_tick = min(tick_times) if tick_times else 0

    # Compute actual FPS from wall clock
    actual_fps = frame_count_actual / max(wall_s, 0.001)

    errors = 0
    for e in event_bus._events if hasattr(event_bus, '_events') else []:
        if "error" in str(e).lower():
            errors += 1

    stability_report = {
        "duration_min": duration_min,
        "backend": backend,
        "target_fps": target_fps,
        "actual_fps": round(actual_fps, 2),
        "frame_count": frame_count_actual,
        "dropped_frames": metrics.get("dropped_frames", 0),
        "unique_tracks": unique_tracks,
        "id_churn_rate": round(id_churn, 4),
        "average_track_quality": metrics.get("average_track_quality", 0),
        "average_tick_ms": round(avg_tick, 2),
        "min_tick_ms": round(min_tick, 2),
        "max_tick_ms": round(max_tick, 2),
        "ram_mb": round(ram_mb, 1),
        "wall_clock_s": round(wall_s, 2),
        "errors": errors,
    }

    report_path = run_dir / "stability_report.json"
    report_path.write_text(json.dumps(stability_report, indent=2), encoding="utf-8")

    logger.info(
        "Stability test complete: {} frames, {:.1f} fps, {} unique tracks, churn={:.4f}, ram={:.0f}MB",
        frame_count_actual, actual_fps, unique_tracks, id_churn, ram_mb,
    )

    return stability_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-run stability test")
    parser.add_argument("--duration-min", type=float, default=10, help="Test duration in minutes")
    parser.add_argument("--backend", default="mock", choices=["mock", "yolo"], help="Perception backend")
    parser.add_argument("--video-source", default="", help="Video file path (empty = synthetic)")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target FPS")
    parser.add_argument("--save-frames", action="store_true", default=True)
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    parser.add_argument("--report-every-s", type=float, default=30.0, help="Progress report interval")
    parser.add_argument("--config", default=None, help="Config YAML path")
    args = parser.parse_args()

    report = run_stability_test(
        duration_min=args.duration_min,
        backend=args.backend,
        video_source=args.video_source,
        target_fps=args.target_fps,
        save_frames=args.save_frames,
        report_every_s=args.report_every_s,
        config_path=args.config,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
