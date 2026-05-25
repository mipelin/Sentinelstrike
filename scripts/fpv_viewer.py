#!/usr/bin/env python3
"""FPV camera viewer — subscribes to Gazebo camera topic and displays via OpenCV."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np


def main() -> None:
    import gz.transport13 as gzt
    import gz.msgs10.image_pb2

    node = gzt.Node()
    latest: list[np.ndarray | None] = [None]

    def on_image(msg: object) -> None:
        if msg.width <= 0 or msg.height <= 0:
            return
        channels = msg.step // msg.width if msg.width > 0 else 3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        if msg.pixel_format_type == 3:  # RGB_INT8
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        latest[0] = arr

    topics = [t for t in node.topic_list() if "image" in t]
    if not topics:
        print("No camera topics found. Is Gazebo running with a camera model?")
        sys.exit(1)

    topic = topics[0]
    print(f"Subscribing to: {topic}")
    node.subscribe(msg_type=gz.msgs10.image_pb2.Image, topic=topic, callback=on_image)
    print("FPV viewer running. Press Q to quit.")

    while True:
        time.sleep(0.03)
        frame = latest[0]
        if frame is not None:
            cv2.imshow("FPV", frame)
            latest[0] = None
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
