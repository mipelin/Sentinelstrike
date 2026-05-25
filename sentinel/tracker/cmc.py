"""Camera Motion Compensation (CMC) for UAV ISR tracking.

Estimates the global motion between consecutive frames using the
Enhanced Correlation Coefficient (ECC) algorithm. The resulting affine
warp is applied to Kalman predicted track positions so that association
works in a camera-stabilized coordinate frame.

This is the single most important component for stable aerial tracking.
Without CMC, any drone movement causes all tracks to be lost because
IoU drops to zero when the camera shifts.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


class CameraMotionCompensator:
    """Estimates and applies frame-to-frame affine warp for CMC."""

    def __init__(
        self,
        enabled: bool = True,
        downscale_width: int = 640,
        ecc_iterations: int = 50,
        ecc_epsilon: float = 0.001,
        motion_threshold: float = 50.0,
    ) -> None:
        self._enabled = enabled
        self._downscale_width = downscale_width
        self._ecc_iterations = ecc_iterations
        self._ecc_epsilon = ecc_epsilon
        self._motion_threshold = motion_threshold
        self._prev_gray: np.ndarray | None = None
        self._warp: np.ndarray = np.eye(2, 3, dtype=np.float64)
        self._scale_factor: float = 1.0
        self._cumulative_warp = np.eye(2, 3, dtype=np.float64)

    @property
    def warp_matrix(self) -> np.ndarray:
        """Current 2x3 affine warp matrix."""
        return self._warp.copy()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Compute warp from previous frame to current frame.

        Returns the 2x3 affine warp matrix (identity if disabled or failed).
        """
        if not self._enabled or not _CV2:
            self._warp = np.eye(2, 3, dtype=np.float64)
            return self._warp

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        # Downscale for speed
        h, w = gray.shape[:2]
        if self._downscale_width > 0 and w > self._downscale_width:
            self._scale_factor = self._downscale_width / w
            gray_small = cv2.resize(
                gray, (self._downscale_width, int(h * self._scale_factor)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            self._scale_factor = 1.0
            gray_small = gray

        if self._prev_gray is None:
            self._prev_gray = gray_small
            self._warp = np.eye(2, 3, dtype=np.float64)
            self._cumulative_warp = np.eye(2, 3, dtype=np.float64)
            return self._warp

        # Estimate affine transform using ECC
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(
                self._prev_gray, gray_small, warp_matrix,
                cv2.MOTION_AFFINE,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self._ecc_iterations,
                    self._ecc_epsilon,
                ),
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error:
            # ECC can fail on uniform frames or extreme motion
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Scale warp back to full resolution
        if self._scale_factor != 1.0:
            warp_matrix[0, 2] /= self._scale_factor
            warp_matrix[1, 2] /= self._scale_factor

        self._warp = warp_matrix.astype(np.float64)
        self._prev_gray = gray_small

        # Update cumulative warp for long-term motion tracking
        self._update_cumulative()

        # If motion is too extreme (scene change), reset
        translation = np.sqrt(self._warp[0, 2] ** 2 + self._warp[1, 2] ** 2)
        if translation > self._motion_threshold:
            self.reset()

        return self._warp

    def _update_cumulative(self) -> None:
        """Accumulate warp for long-term motion estimation."""
        new_cum = np.zeros((2, 3), dtype=np.float64)
        new_cum[:2, :2] = self._warp[:2, :2] @ self._cumulative_warp[:2, :2]
        new_cum[:2, 2] = self._warp[:2, :2] @ self._cumulative_warp[:2, 2] + self._warp[:2, 2]
        self._cumulative_warp = new_cum

    def warp_point(self, x: float, y: float) -> tuple[float, float]:
        """Apply current warp to a 2D point."""
        pt = np.array([x, y, 1.0])
        result = self._warp @ pt
        return float(result[0]), float(result[1])

    def warp_bbox(
        self, bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Apply current warp to a bounding box.

        Warps the center, keeps size unchanged (affine doesn't distort much).
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        ncx, ncy = self.warp_point(cx, cy)
        hw = (x2 - x1) / 2.0
        hh = (y2 - y1) / 2.0
        return (ncx - hw, ncy - hh, ncx + hw, ncy + hh)

    def reset(self) -> None:
        """Reset CMC state (use after scene change)."""
        self._prev_gray = None
        self._warp = np.eye(2, 3, dtype=np.float64)
        self._cumulative_warp = np.eye(2, 3, dtype=np.float64)
