import math
from typing import Iterable, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return math.hypot(dx, dy)

def angle_degrees(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Angle from p1 to p2 in degrees."""
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return math.degrees(math.atan2(dy, dx))

def normalize_value(value: float, scale: float) -> float:
    return value / max(scale, 1e-6)

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def midpoint(p1: Sequence[float], p2: Sequence[float]) -> Point:
    return ((float(p1[0]) + float(p2[0])) * 0.5, (float(p1[1]) + float(p2[1])) * 0.5)

def pinch_ratio(thumb_tip: Sequence[float], index_tip: Sequence[float], hand_scale: float) -> float:
    """Normalized pinch distance relative to a hand scale baseline."""
    return normalize_value(distance(thumb_tip, index_tip), hand_scale)

def to_numpy(point: Sequence[float] | Iterable[float]) -> np.ndarray:
    return np.asarray(point, dtype=float)

# --- Geometry utilities for stroke simplification ---

def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = clamp(t, 0.0, 1.0)
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def rdp(points: Sequence[Sequence[float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer–Douglas–Peucker simplification."""
    if len(points) < 3:
        return [(float(p[0]), float(p[1])) for p in points]

    start, end = points[0], points[-1]
    max_dist = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        dist = _point_segment_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            idx = i

    if max_dist > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]

