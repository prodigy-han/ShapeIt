import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

SMOOTHING = 0.35  # lower = more smoothing


def _normalize_angle(delta: float) -> float:
    """Normalize an angle delta to [-180, 180] to smooth wraparound."""
    if delta > 180:
        delta -= 360
    if delta < -180:
        delta += 360
    return delta


ColorBGR = Tuple[int, int, int]
Point = Tuple[float, float]


@dataclass
class Shape2D:
    center: np.ndarray
    size: float
    color: ColorBGR = (255, 100, 50)
    angle: float = 0.0
    target_center: np.ndarray = field(default=None)
    target_size: float = None
    target_angle: float = 0.0
    is_selected: bool = False

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        self.target_center = np.asarray(
            self.target_center if self.target_center is not None else self.center,
            dtype=float,
        )
        self.size = float(self.size)
        self.target_size = float(self.target_size) if self.target_size is not None else float(self.size)
        self.angle = float(self.angle)
        self.target_angle = float(self.target_angle)

    def contains_point(self, point: Sequence[float], rotated: bool = True) -> bool:
        """Check if a point is inside the shape. Supports rotated hit-test."""
        px, py = float(point[0]), float(point[1])
        cx, cy = self.center
        dx, dy = px - cx, py - cy
        if rotated and self.angle:
            theta = math.radians(-self.angle)
            cos_a, sin_a = math.cos(theta), math.sin(theta)
            dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
        half = self.size / 2.0
        return -half <= dx <= half and -half <= dy <= half

    def update(self) -> None:
        """Move properties toward their targets with smoothing."""
        self.center += (self.target_center - self.center) * SMOOTHING
        self.size += (self.target_size - self.size) * SMOOTHING

        angle_delta = _normalize_angle(self.target_angle - self.angle)
        self.angle = (self.angle + angle_delta * SMOOTHING) % 360.0

