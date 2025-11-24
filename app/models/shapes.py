import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

SMOOTHING = 0.35  # lower = more smoothing
SEGMENT_TOLERANCE = 8.0


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


@dataclass
class PolylineShape:
    points: list[np.ndarray]
    color: ColorBGR = (0, 165, 255)
    thickness: int = 3
    tolerance: float = SEGMENT_TOLERANCE
    is_selected: bool = False

    def update(self) -> None:
        # No smoothing needed for static polylines.
        return

    def contains_point(self, point: Sequence[float]) -> bool:
        px, py = float(point[0]), float(point[1])
        if len(self.points) < 2:
            return False
        for i in range(len(self.points) - 1):
            if _point_segment_distance((px, py), self.points[i], self.points[i + 1]) <= self.tolerance:
                return True
        return False


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)
