from __future__ import annotations

from typing import Dict, Sequence

from app.input.features import pinch_ratio as compute_pinch_ratio

PINCH_START = 0.35
PINCH_END = 0.45


class GestureInterpreter:
    """Gesture rule engine with pinch hysteresis."""

    def __init__(self, pinch_start: float = PINCH_START, pinch_end: float = PINCH_END) -> None:
        self.pinch_start = pinch_start
        self.pinch_end = pinch_end
        self._is_pinching = False

    @staticmethod
    def _finger_up(points: Sequence[Sequence[float]], tip_idx: int, pip_idx: int) -> bool:
        # In image coordinates, lower y means higher on the hand.
        return points[tip_idx][1] < points[pip_idx][1]

    def compute(self, points: Sequence[Sequence[float]], hand_scale: float) -> Dict[str, float | bool]:
        pinch_ratio = compute_pinch_ratio(points[4], points[8], hand_scale)

        if not self._is_pinching and pinch_ratio < self.pinch_start:
            self._is_pinching = True
        elif self._is_pinching and pinch_ratio > self.pinch_end:
            self._is_pinching = False

        index_up = self._finger_up(points, 8, 6)
        middle_up = self._finger_up(points, 12, 10)
        ring_up = self._finger_up(points, 16, 14)
        pinky_up = self._finger_up(points, 20, 18)
        # Thumb: simple horizontal check for right hand assumption.
        thumb_open = points[4][0] > points[3][0]

        is_two_finger_point = index_up and middle_up and not ring_up and not pinky_up
        is_open_palm = index_up and middle_up and ring_up and pinky_up
        is_pointing = index_up and not middle_up and not ring_up and not pinky_up

        return {
            "pinch_ratio": pinch_ratio,
            "is_pinching": self._is_pinching,
            "is_two_finger_point": is_two_finger_point,
            "is_open_palm": is_open_palm,
            "is_pointing": is_pointing,
            "thumb_open": thumb_open,
        }

