import math
import time
from typing import Any

import numpy as np


class OneEuroFilter:
    """Self-contained One Euro filter for smoothing noisy signals."""

    def __init__(self, freq: float = 120.0, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._last_time: float | None = None
        self._prev_value: np.ndarray | None = None
        self._prev_derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: Any, timestamp: float | None = None) -> Any:
        now = timestamp if timestamp is not None else time.time()
        if self._last_time is None:
            self._last_time = now
            self._prev_value = np.asarray(value, dtype=float)
            self._prev_derivative = np.zeros_like(self._prev_value)
            return value

        dt = max(1e-6, now - self._last_time)
        self._last_time = now

        value_arr = np.asarray(value, dtype=float)
        prev = self._prev_value
        prev_derivative = self._prev_derivative

        derivative = (value_arr - prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        derivative_hat = alpha_d * derivative + (1.0 - alpha_d) * prev_derivative

        cutoff = self.min_cutoff + self.beta * np.abs(derivative_hat)
        alpha = self._alpha(cutoff, dt)
        filtered = alpha * value_arr + (1.0 - alpha) * prev

        self._prev_value = filtered
        self._prev_derivative = derivative_hat
        return filtered

