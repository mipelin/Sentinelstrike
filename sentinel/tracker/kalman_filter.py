"""8-state linear Kalman filter for bounding box tracking.

State vector: [cx, cy, a, h, vcx, vcy, va, vh]
  cx, cy = bounding box center
  a      = aspect ratio (width / height)
  h      = height
  vcx...vh = velocities of the above

Measurement vector: [cx, cy, a, h]

Standard DeepSORT / BoT-SORT state representation.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Linear constant-velocity Kalman filter for bbox tracking."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        self._x = np.zeros(8, dtype=np.float64)
        cx, cy, w, h = self._bbox_to_z(bbox)
        self._x[:4] = [cx, cy, w / max(h, 1.0), h]

        self._P = np.eye(8, dtype=np.float64)
        self._P[4:, 4:] *= 1000.0  # high uncertainty on velocities
        self._P[:4, :4] *= 10.0

        self._q = process_noise
        self._r = measurement_noise
        self._dt = 1.0

    @staticmethod
    def _bbox_to_z(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h

    @staticmethod
    def _z_to_bbox(cx: float, cy: float, a: float, h: float) -> tuple[float, float, float, float]:
        w = a * h
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def predict(self, dt: float = 1.0) -> tuple[float, float, float, float]:
        """Predict state forward by dt. Returns predicted bbox."""
        self._dt = dt

        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        Q = np.zeros((8, 8), dtype=np.float64)
        for i in range(4):
            Q[i, i] = self._q * dt
            Q[i + 4, i + 4] = self._q * dt * dt
            Q[i, i + 4] = self._q * dt * 0.5
            Q[i + 4, i] = self._q * dt * 0.5

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        return self._state_to_bbox()

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        """Update state with a measurement bbox."""
        cx, cy, w, h = self._bbox_to_z(bbox)
        a = w / max(h, 1.0)
        z = np.array([cx, cy, a, h], dtype=np.float64)

        H = np.zeros((4, 8), dtype=np.float64)
        H[:4, :4] = np.eye(4)

        R = np.eye(4, dtype=np.float64) * self._r
        R[2, 2] *= 10.0  # aspect ratio is noisier
        R[3, 3] *= 10.0  # height is noisier

        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x = self._x + K @ y
        self._P = (np.eye(8) - K @ H) @ self._P

    def get_state_bbox(self) -> tuple[float, float, float, float]:
        """Current state as bbox (x1, y1, x2, y2)."""
        return self._state_to_bbox()

    def get_velocity(self) -> tuple[float, float]:
        """Current velocity (vcx, vcy) in pixels/frame."""
        return float(self._x[4]), float(self._x[5])

    def get_speed(self) -> float:
        """Current speed in pixels/frame."""
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx * vx + vy * vy))

    def apply_warp(self, warp_matrix: np.ndarray) -> None:
        """Apply 2x3 affine warp to predicted position (CMC).

        Warps (cx, cy) through the affine transform, leaving a, h unchanged.
        """
        cx, cy = self._x[0], self._x[1]
        pt = np.array([cx, cy, 1.0])
        new_pt = warp_matrix @ pt
        self._x[0] = new_pt[0]
        self._x[1] = new_pt[1]

    def get_innovation(self, bbox: tuple[float, float, float, float]) -> float:
        """Mahalanobis-like distance between prediction and measurement.

        Used for gating: low distance = good match candidate.
        """
        cx, cy, w, h = self._bbox_to_z(bbox)
        a = w / max(h, 1.0)
        z = np.array([cx, cy, a, h], dtype=np.float64)

        H = np.zeros((4, 8), dtype=np.float64)
        H[:4, :4] = np.eye(4)

        y = z - H @ self._x
        S = H @ self._P @ H.T + np.eye(4) * self._r
        return float(np.sqrt(y @ np.linalg.inv(S) @ y))

    def _state_to_bbox(self) -> tuple[float, float, float, float]:
        cx = float(self._x[0])
        cy = float(self._x[1])
        a = float(max(self._x[2], 0.01))
        h = float(max(self._x[3], 1.0))
        return self._z_to_bbox(cx, cy, a, h)
