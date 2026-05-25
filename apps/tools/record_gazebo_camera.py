"""Record Gazebo camera feed to video file or JPEG sequence.

Usage:
    # Record MP4 for 60 seconds:
    python -m apps.tools.record_gazebo_camera \
        --topic /world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image \
        --output /tmp/gazebo_recording.mp4 \
        --duration 60

    # Record with frame stamping:
    python -m apps.tools.record_gazebo_camera --topic ... --output out.mp4 --stamp-frames

    # Save sequential JPEGs for dataset generation:
    python -m apps.tools.record_gazebo_camera --topic ... --save-jpegs /tmp/frames --duration 30
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

try:
    import gz.transport13 as _gzt
    import gz.msgs10.image_pb2 as _image_pb2

    _GZ_AVAILABLE = True
except ImportError:
    _GZ_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# Reuse pixel format mapping from view_gazebo_camera
from apps.tools.view_gazebo_camera import FrameState, _msg_to_bgr


def _stamp_frame(frame: np.ndarray, frame_id: int, fps: float, t_elapsed: float) -> np.ndarray:
    """Add timestamp overlay to frame. Returns a copy."""
    out = frame.copy()
    h, w = out.shape[:2]
    lines = [
        f"#{frame_id}  {fps:.1f} FPS  T+{t_elapsed:.1f}s",
        f"{w}x{h}  {time.strftime('%H:%M:%S')}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    for i, line in enumerate(lines):
        (tw, th), baseline = cv2.getTextSize(line, font, scale, thickness)
        y0 = out.shape[0] - (len(lines) - i) * (th + baseline + 6)
        cv2.rectangle(
            out, (4, y0 - th - 4), (tw + 10, y0 + baseline + 2),
            (0, 0, 0), cv2.FILLED,
        )
        cv2.putText(out, line, (6, y0), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Gazebo camera feed")
    parser.add_argument("--topic", required=True, help="Gazebo image topic")
    parser.add_argument("--output", help="Output video file (MP4/AVI)")
    parser.add_argument("--save-jpegs", metavar="DIR", help="Save sequential JPEGs to directory")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds (default: 60)")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target capture FPS (default: 10)")
    parser.add_argument("--stamp-frames", action="store_true", help="Add frame ID/timestamp overlay")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for first frame")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec (default: mp4v)")
    args = parser.parse_args()

    if not _GZ_AVAILABLE:
        print("ERROR: gz.transport13 / gz.msgs10 not available", file=sys.stderr)
        sys.exit(1)

    if not _CV2_AVAILABLE:
        print("ERROR: opencv-python not available", file=sys.stderr)
        sys.exit(1)

    if not args.output and not args.save_jpegs:
        parser.error("Specify --output or --save-jpegs")

    state = FrameState()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # Need threading import
    import threading

    print(f"Subscribing to: {args.topic}")
    node = _gzt.Node()
    node.subscribe(msg_type=_image_pb2.Image, topic=args.topic, callback=state.on_msg)

    print(f"Waiting up to {args.timeout}s for first frame...")
    if not state.connected.wait(timeout=args.timeout):
        print(f"FAIL: no frame within {args.timeout}s", file=sys.stderr)
        sys.exit(2)

    # Get frame dimensions from first frame
    first_frame, _ = state.snapshot()
    if first_frame is None:
        print("FAIL: first frame is None", file=sys.stderr)
        sys.exit(2)

    h, w = first_frame.shape[:2]
    print(f"Connected: {w}x{h} @ target {args.target_fps} FPS for {args.duration}s")

    # Setup video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        writer = cv2.VideoWriter(args.output, fourcc, args.target_fps, (w, h))
        if not writer.isOpened():
            print(f"ERROR: cannot open {args.output} for writing", file=sys.stderr)
            sys.exit(1)
        print(f"Recording to: {args.output}")

    # Setup JPEG directory
    jpeg_dir = None
    if args.save_jpegs:
        jpeg_dir = Path(args.save_jpegs)
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving JPEGs to: {jpeg_dir}")

    t_start = time.monotonic()
    recorded = 0
    last_fps_time = t_start
    last_fps_count = 0
    current_fps = 0.0
    frame_interval = 1.0 / args.target_fps
    last_capture = 0.0

    while not stop.is_set():
        now = time.monotonic()
        elapsed = now - t_start

        if elapsed >= args.duration:
            break

        if now - last_capture < frame_interval:
            time.sleep(0.005)
            continue

        last_capture = now
        frame, recv_count = state.snapshot()

        if frame is None:
            continue

        # FPS calculation
        dt = now - last_fps_time
        if dt >= 1.0:
            current_fps = (recv_count - last_fps_count) / dt
            last_fps_time = now
            last_fps_count = recv_count

        # Stamp frame if requested
        display = frame.copy()
        if args.stamp_frames:
            display = _stamp_frame(display, recv_count, current_fps, elapsed)

        # Write to video
        if writer is not None:
            writer.write(display)

        # Save JPEG
        if jpeg_dir is not None:
            jpeg_path = jpeg_dir / f"frame_{recorded:06d}.jpg"
            cv2.imwrite(str(jpeg_path), display)

        recorded += 1

        if recorded % 50 == 0:
            print(f"  [{elapsed:.1f}s] recorded: {recorded}  received: {recv_count}  FPS: {current_fps:.1f}")

    if writer is not None:
        writer.release()

    final_elapsed = time.monotonic() - t_start
    _, final_recv = state.snapshot()

    print()
    print(f"Recording complete:")
    print(f"  duration: {final_elapsed:.1f}s (target: {args.duration}s)")
    print(f"  frames recorded: {recorded}")
    print(f"  frames received: {final_recv}")
    print(f"  avg record FPS: {recorded / final_elapsed:.1f}")
    if args.output:
        size_mb = Path(args.output).stat().st_size / (1024 * 1024)
        print(f"  file: {args.output} ({size_mb:.1f} MB)")
    if jpeg_dir:
        jpeg_count = len(list(jpeg_dir.glob("*.jpg")))
        print(f"  jpegs: {jpeg_count} files in {jpeg_dir}")


if __name__ == "__main__":
    main()
