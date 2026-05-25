"""Live Gazebo camera viewer — subscribe to an image topic and display or save frames.

Usage:
    # GUI mode (requires display):
    python -m apps.tools.view_gazebo_camera \
        --topic /world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image

    # Headless mode (save frames instead of display):
    python -m apps.tools.view_gazebo_camera \
        --topic /world/sentinel_street/model/x500_mono_cam_0/link/camera_link/sensor/camera/image \
        --headless --save-latest /tmp/gazebo_camera.jpg

    # Resize for Moonlight/streaming:
    python -m apps.tools.view_gazebo_camera --topic ... --resize-width 1280

    # Debug frame counter:
    python -m apps.tools.view_gazebo_camera --topic ... --headless --save-latest /tmp/cam.jpg --debug-frame-counter

Controls (GUI mode):
    s   — save current frame to /tmp/gazebo_frame_<timestamp>.jpg
    f   — toggle FPS overlay
    q   — quit
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
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

_FORMAT_CHANNELS = {
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

_WINDOW_WIDTH = 960
_WINDOW_HEIGHT = 720


class FrameState:
    """Thread-safe holder for the latest frame and received count.

    Callback writes, consumer loop reads — always via snapshot() for consistency.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._received_count = 0
        self._first_msg_logged = False
        self.connected = threading.Event()

    def on_msg(self, msg: object) -> None:
        bgr = _msg_to_bgr(msg)
        if bgr is None:
            return
        with self._lock:
            self._latest_frame = bgr.copy()
            self._received_count += 1
            if not self._first_msg_logged:
                _log_first_metadata(msg, self._received_count)
                self._first_msg_logged = True
        self.connected.set()

    def snapshot(self) -> tuple[np.ndarray | None, int]:
        """Return (latest_frame_copy, received_count) atomically."""
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            return frame, self._received_count


def _msg_to_bgr(msg: object) -> np.ndarray | None:
    w = msg.width
    h = msg.height
    step = msg.step
    pixel_format = msg.pixel_format_type
    raw = bytes(msg.data)

    if w == 0 or h == 0 or len(raw) == 0:
        return None

    info = _FORMAT_CHANNELS.get(pixel_format)
    if info is None:
        return None

    fmt_name, channels, dtype = info

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


def _log_first_metadata(msg: object, frame_count: int) -> None:
    w, h = msg.width, msg.height
    pf = msg.pixel_format_type
    fmt_name = _FORMAT_CHANNELS.get(pf, (f"UNKNOWN({pf})",))[0]
    stamp_sec = getattr(msg.header, "stamp", None)
    ts = ""
    if stamp_sec is not None:
        sec = getattr(stamp_sec, "sec", 0)
        nsec = getattr(stamp_sec, "nsec", 0)
        ts = f"  stamp: {sec}.{nsec:09d}s"
    print(f"[{frame_count}] First frame metadata:")
    print(f"  resolution: {w}x{h}")
    print(f"  pixel_format: {fmt_name} (enum={pf})")
    print(f"  step: {msg.step}  data: {len(msg.data)} bytes{ts}")


def _apply_transform(frame: np.ndarray, rotate: int, flip: str) -> np.ndarray:
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip == "h":
        frame = cv2.flip(frame, 1)
    elif flip == "v":
        frame = cv2.flip(frame, 0)
    elif flip == "hv":
        frame = cv2.flip(frame, -1)
    return frame


def _resize(frame: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def _overlay_text(
    frame: np.ndarray, fps: float, frame_count: int, timestamp: float
) -> np.ndarray:
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}  {w}x{h}  #{frame_count}  T+{timestamp:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad = 4
    x0, y0 = 6, frame.shape[0] - th - pad - baseline - 4
    cv2.rectangle(
        frame, (x0, y0 - pad), (x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad),
        (0, 0, 0), cv2.FILLED,
    )
    cv2.putText(frame, text, (x0 + pad, y0 + th), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return frame


def _run_headless(
    args: argparse.Namespace,
    state: FrameState,
    stop_event: threading.Event,
    t_start: float,
) -> None:
    last_save = 0.0
    save_path = Path(args.save_latest) if args.save_latest else None
    saved_every_set: set[int] = set()
    displayed_count = 0
    last_debug_time = t_start

    while not stop_event.is_set():
        frame, recv_count = state.snapshot()
        now = time.time()

        if frame is not None:
            displayed_count += 1
            frame = _apply_transform(frame, args.rotate, args.flip if args.flip != "none" else "")

            # Periodic save-latest
            if save_path and (now - last_save) >= args.save_interval:
                cv2.imwrite(str(save_path), frame)
                last_save = now
                elapsed = time.monotonic() - t_start
                fps = recv_count / elapsed if elapsed > 0 else 0
                print(f"[{recv_count}] {fps:.1f} FPS — saved to {save_path} ({frame.shape[1]}x{frame.shape[0]})")

            # Save every Nth frame
            if args.save_every > 0 and recv_count % args.save_every == 0 and recv_count not in saved_every_set:
                path = f"/tmp/gazebo_frame_{recv_count:06d}.jpg"
                cv2.imwrite(path, frame)
                saved_every_set.add(recv_count)

        # Debug frame counter — once per second
        if args.debug_frame_counter and (time.monotonic() - last_debug_time) >= 1.0:
            _, rc = state.snapshot()
            print(f"  DEBUG: received={rc}  displayed={displayed_count}  saved_every={len(saved_every_set)}")
            last_debug_time = time.monotonic()

        if args.max_frames > 0 and recv_count >= args.max_frames:
            print(f"Reached {args.max_frames} frames. Done.")
            break

        time.sleep(0.05)


def _run_gui(
    args: argparse.Namespace,
    state: FrameState,
    stop_event: threading.Event,
    t_start: float,
) -> None:
    window_name = "Gazebo Camera  [s]save [f]overlay [q]quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, _WINDOW_WIDTH, _WINDOW_HEIGHT)

    show_overlay = not args.no_overlay
    last_save_time = 0.0
    last_fps_time = time.monotonic()
    last_fps_count = 0
    current_fps = 0.0
    saved_every_set: set[int] = set()
    displayed_count = 0
    last_debug_time = t_start

    flip_code = args.flip if args.flip != "none" else ""

    while not stop_event.is_set():
        frame, recv_count = state.snapshot()

        if frame is not None:
            displayed_count += 1
            display = _apply_transform(frame, args.rotate, flip_code)
            display = _resize(display, args.resize_width if args.resize_width > 0 else None)

            # FPS calculation (update every 0.5s)
            now_mono = time.monotonic()
            dt = now_mono - last_fps_time
            if dt >= 0.5:
                current_fps = (recv_count - last_fps_count) / dt
                last_fps_time = now_mono
                last_fps_count = recv_count

            elapsed = now_mono - t_start

            if show_overlay:
                display = _overlay_text(display, current_fps, recv_count, elapsed)

            cv2.imshow(window_name, display)

            # Save-every-Nth (auto-save sequence)
            if args.save_every > 0 and recv_count not in saved_every_set and recv_count % args.save_every == 0:
                cv2.imwrite(f"/tmp/gazebo_frame_{recv_count:06d}.jpg", frame)
                saved_every_set.add(recv_count)

        key = cv2.pollKey()
        if key == ord("q"):
            break
        elif key == ord("s") and frame is not None:
            now = time.time()
            if now - last_save_time >= 0.3:
                _, rc = state.snapshot()
                path = f"/tmp/gazebo_frame_{int(now)}.jpg"
                cv2.imwrite(path, frame)
                print(f"[{rc}] Saved: {path}")
                last_save_time = now
        elif key == ord("f"):
            show_overlay = not show_overlay

        # Debug frame counter — once per second
        if args.debug_frame_counter and (time.monotonic() - last_debug_time) >= 1.0:
            _, rc = state.snapshot()
            print(f"  DEBUG: received={rc}  displayed={displayed_count}  saved_every={len(saved_every_set)}")
            last_debug_time = time.monotonic()

        if args.max_frames > 0 and recv_count >= args.max_frames:
            print(f"Reached {args.max_frames} frames. Done.")
            break

        time.sleep(0.005)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="View Gazebo camera feed in real time")
    parser.add_argument(
        "--topic",
        required=True,
        help="Gazebo image topic",
    )
    parser.add_argument("--headless", action="store_true", help="No GUI window, use with --save-latest")
    parser.add_argument("--save-latest", metavar="PATH", help="Continuously save latest frame to this path")
    parser.add_argument(
        "--save-interval",
        type=float,
        default=1.0,
        help="Min seconds between --save-latest writes (default: 1.0)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save every Nth frame to /tmp/gazebo_frame_NNNN.jpg (0 = disabled)",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for first frame (default: 15)")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = unlimited)")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Resize display to this width in pixels (0 = native). Useful for Moonlight streaming.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate image by degrees (default: 0)",
    )
    parser.add_argument(
        "--flip",
        type=str,
        default="none",
        choices=["none", "h", "v", "hv"],
        help="Flip image: h=horizontal, v=vertical, hv=both (default: none)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable FPS/resolution overlay in GUI mode",
    )
    parser.add_argument(
        "--debug-frame-counter",
        action="store_true",
        help="Print received/displayed/saved counts once per second",
    )
    args = parser.parse_args()

    if not _GZ_AVAILABLE:
        print("ERROR: gz.transport13 / gz.msgs10 not available", file=sys.stderr)
        print("  Install: sudo apt install python3-gz-transport13 python3-gz-msgs10", file=sys.stderr)
        sys.exit(1)

    if not _CV2_AVAILABLE:
        print("ERROR: opencv-python not available. Install opencv-python.", file=sys.stderr)
        sys.exit(1)

    state = FrameState()
    stop_event = threading.Event()

    # Subscribe
    print(f"Subscribing to: {args.topic}")
    node = _gzt.Node()
    node.subscribe(
        msg_type=_image_pb2.Image,
        topic=args.topic,
        callback=state.on_msg,
    )

    print(f"Waiting up to {args.timeout}s for first frame...")
    if not state.connected.wait(timeout=args.timeout):
        print(f"FAIL: no frame received within {args.timeout}s. Is Gazebo running with this topic?", file=sys.stderr)
        sys.exit(2)

    print("Connected! Receiving frames...")
    t_start = time.monotonic()

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    if args.headless:
        _run_headless(args, state, stop_event, t_start)
    else:
        _run_gui(args, state, stop_event, t_start)

    _, final_count = state.snapshot()
    print(f"Total frames received: {final_count}")


if __name__ == "__main__":
    main()
