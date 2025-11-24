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

