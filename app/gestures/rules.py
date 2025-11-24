from __future__ import annotations

from typing import Dict, Sequence

from app.input.features import pinch_ratio as compute_pinch_ratio

PINCH_START = 0.35
PINCH_END = 0.45
DRAW_START_FRAMES = 3
DRAW_END_FRAMES = 3
EXIT_FRAMES = 3


class DebouncedFlag:
    """Generic debounce helper for a boolean signal."""

    def __init__(self, on_frames: int, off_frames: int) -> None:
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.on_count = 0
        self.off_count = 0
        self.is_active = False

    def update(self, raw_on: bool) -> bool:
        if raw_on:
            self.on_count += 1
            self.off_count = 0
            if not self.is_active and self.on_count >= self.on_frames:
                self.is_active = True
        else:
            self.off_count += 1
            self.on_count = 0
            if self.is_active and self.off_count >= self.off_frames:
                self.is_active = False
        return self.is_active


class DrawGestureState:
    """Debounce draw gesture on/off transitions."""

    def __init__(self, start_frames: int = DRAW_START_FRAMES, end_frames: int = DRAW_END_FRAMES) -> None:
        self.flag = DebouncedFlag(on_frames=start_frames, off_frames=end_frames)

    def update(self, is_raw_on: bool) -> bool:
        return self.flag.update(is_raw_on)


class GestureInterpreter:
    """Gesture rule engine with pinch hysteresis."""

    def __init__(self, pinch_start: float = PINCH_START, pinch_end: float = PINCH_END) -> None:
        self.pinch_start = pinch_start
        self.pinch_end = pinch_end
        self._is_pinching = False
        self._draw_state = DrawGestureState()
        self._exit_state = DebouncedFlag(on_frames=EXIT_FRAMES, off_frames=EXIT_FRAMES)

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
        is_three_finger = index_up and middle_up and ring_up and not pinky_up
        is_thumb_up = thumb_open and not (index_up or middle_up or ring_up or pinky_up)
        is_draw_gesture = is_three_finger
        is_draw_active = self._draw_state.update(is_draw_gesture)
        all_fingers_down = not (index_up or middle_up or ring_up or pinky_up)
        is_fist = all_fingers_down
        is_exit_gesture = self._exit_state.update(is_fist)

        return {
            "pinch_ratio": pinch_ratio,
            "is_pinching": self._is_pinching,
            "is_two_finger_point": is_two_finger_point,
            "is_open_palm": is_open_palm,
            "is_pointing": is_pointing,
            "thumb_open": thumb_open,
            "is_three_finger": is_three_finger,
            "is_thumb_up": is_thumb_up,
            "is_draw_gesture": is_draw_gesture,
            "is_draw_active": is_draw_active,
            "is_exit_gesture": is_exit_gesture,
        }
