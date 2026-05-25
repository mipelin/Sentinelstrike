"""PID controller with safety clamps for UAV yaw follow.

Designed for image-space error → yaw_rate conversion.
Includes deadband, integral anti-windup, output clamping, and derivative filtering.
"""

from __future__ import annotations


class PIDController:
    """Discrete PID controller with safety features."""

    def __init__(
        self,
        kp: float = 0.005,
        ki: float = 0.0001,
        kd: float = 0.002,
        deadband: float = 30.0,
        output_min: float = -0.5,
        output_max: float = 0.5,
        integral_min: float = -2.0,
        integral_max: float = 2.0,
        derivative_filter: float = 0.7,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._deadband = deadband
        self._output_min = output_min
        self._output_max = output_max
        self._integral_min = integral_min
        self._integral_max = integral_max
        self._derivative_filter = derivative_filter

        self._integral: float = 0.0
        self._prev_error: float | None = None
        self._filtered_derivative: float = 0.0

    def update(self, error: float, dt: float = 1.0) -> float:
        """Compute PID output for given error.

        Args:
            error: Image-space error in pixels (positive = target right of center)
            dt: Time delta since last update (normalized frames, default 1.0)

        Returns:
            Control output (yaw rate in rad/s)
        """
        # Deadband — ignore small errors to prevent jitter
        if abs(error) < self._deadband:
            error = 0.0
            self._integral *= 0.9  # slowly decay integral in deadband

        # Proportional
        p_term = self._kp * error

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = max(self._integral_min, min(self._integral_max, self._integral))
        i_term = self._ki * self._integral

        # Derivative with low-pass filter
        if self._prev_error is not None and dt > 0:
            raw_derivative = (error - self._prev_error) / dt
            self._filtered_derivative = (
                self._derivative_filter * raw_derivative
                + (1 - self._derivative_filter) * self._filtered_derivative
            )
            d_term = self._kd * self._filtered_derivative
        else:
            d_term = 0.0

        self._prev_error = error

        output = p_term + i_term + d_term
        return max(self._output_min, min(self._output_max, output))

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0

    @property
    def state(self) -> dict:
        """Current PID state for diagnostics."""
        return {
            "integral": round(self._integral, 4),
            "prev_error": round(self._prev_error, 1) if self._prev_error is not None else None,
            "filtered_derivative": round(self._filtered_derivative, 4),
        }
