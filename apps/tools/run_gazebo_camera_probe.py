"""Diagnostic CLI — probe a Gazebo image topic, print metadata, save one frame.

Usage:
    python -m apps.tools.run_gazebo_camera_probe --topic <topic> --timeout 10 --save-frame /tmp/gz_frame.jpg
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

try:
    import gz.transport13 as _gzt
    import gz.msgs10.image_pb2 as _image_pb2

    _GZ_AVAILABLE = True
except ImportError:
    _GZ_AVAILABLE = False

_FORMAT_NAMES = {
    3: "RGB_INT8",
    4: "RGBA_INT8",
    5: "BGRA_INT8",
    6: "RGB_INT16",
    7: "RGB_INT32",
    8: "BGR_INT8",
    9: "R8G8B8",
    10: "RGB16",
    29: "L_INT8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a Gazebo image topic")
    parser.add_argument("--topic", required=True, help="Gazebo image topic to subscribe to")
    parser.add_argument("--timeout", type=float, default=10.0, help="Max seconds to wait for first message")
    parser.add_argument("--save-frame", default=None, help="Path to save first frame as JPG")
    args = parser.parse_args()

    if not _GZ_AVAILABLE:
        print("ERROR: gz.transport13 / gz.msgs10 not available", file=sys.stderr)
        print("  Install: python3-gz-transport13 python3-gz-msgs10", file=sys.stderr)
        sys.exit(1)

    msg_received = threading.Event()
    msg_holder: list[object] = []

    def _on_msg(msg: object) -> None:
        if not msg_received.is_set():
            msg_holder.append(msg)
            msg_received.set()

    print(f"Subscribing to: {args.topic}")
    node = _gzt.Node()
    node.subscribe(
        msg_type=_image_pb2.Image,
        topic=args.topic,
        callback=_on_msg,
    )

    print(f"Waiting up to {args.timeout}s for a message...")
    if not msg_received.wait(timeout=args.timeout):
        print(f"FAIL: no message received within {args.timeout}s", file=sys.stderr)
        sys.exit(2)

    msg = msg_holder[0]
    fmt_type = msg.pixel_format_type
    fmt_name = _FORMAT_NAMES.get(fmt_type, f"UNKNOWN({fmt_type})")

    print(f"Message type: gz.msgs10.image_pb2.Image")
    print(f"  width:          {msg.width}")
    print(f"  height:         {msg.height}")
    print(f"  pixel_format:   {fmt_name} (enum={fmt_type})")
    print(f"  step:           {msg.step}")
    print(f"  data length:    {len(msg.data)} bytes")

    if args.save_frame:
        import cv2

        w = msg.width
        h = msg.height
        step = msg.step
        pixel_format = msg.pixel_format_type
        raw = bytes(msg.data)

        if w == 0 or h == 0 or len(raw) == 0:
            print(f"FAIL: image has zero dimensions or empty data", file=sys.stderr)
            sys.exit(3)

        channels = step // w if w > 0 else 3
        dtype = np.uint8
        if pixel_format in (6, 9):
            dtype = np.uint16
            channels = step // (w * 2)
        elif pixel_format in (7, 10):
            dtype = np.uint32
            channels = step // (w * 4)

        if pixel_format == 29:  # L_INT8
            grey = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
            frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        else:
            frame = np.frombuffer(raw, dtype=dtype).reshape((h, w, channels))
            if pixel_format == 3:  # RGB_INT8
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif pixel_format == 5:  # BGRA_INT8
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif pixel_format == 4:  # RGBA_INT8
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif pixel_format == 8:  # BGR_INT8 — already BGR
                pass

        success = cv2.imwrite(args.save_frame, frame)
        if success:
            print(f"  Saved frame to: {args.save_frame}")
        else:
            print(f"FAIL: cv2.imwrite failed for {args.save_frame}", file=sys.stderr)
            sys.exit(4)

    print("OK")


if __name__ == "__main__":
    main()
