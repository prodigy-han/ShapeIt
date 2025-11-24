from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp

from .features import distance
from .filters import OneEuroFilter

Point = Tuple[float, float]


class HandTracker:
    """Wrapper around MediaPipe Hands that outputs pixel landmarks with smoothing."""

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
        filter_kwargs: Optional[dict] = None,
    ) -> None:
        filter_kwargs = filter_kwargs or {"min_cutoff": 1.5, "beta": 0.02}
        self._filters: List[Tuple[OneEuroFilter, OneEuroFilter]] = []
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._filter_kwargs = filter_kwargs
        self._last_landmarks = None

    def _ensure_filters(self, count: int) -> None:
        while len(self._filters) < count:
            self._filters.append(
                (OneEuroFilter(**self._filter_kwargs), OneEuroFilter(**self._filter_kwargs))
            )

    def _apply_filters(self, points: List[Point], timestamp: float) -> List[Point]:
        self._ensure_filters(len(points))
        filtered: List[Point] = []
        for idx, (x, y) in enumerate(points):
            fx_filter, fy_filter = self._filters[idx]
            fx = fx_filter(x, timestamp=timestamp)
            fy = fy_filter(y, timestamp=timestamp)
            filtered.append((float(fx), float(fy)))
        return filtered

    def get_landmarks(self, frame, timestamp: Optional[float] = None) -> Optional[Dict]:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            self._last_landmarks = None
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        raw_points = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
        hand_scale = distance(raw_points[0], raw_points[5])  # wrist to index MCP
        timestamp = timestamp if timestamp is not None else time.time()
        filtered_points = self._apply_filters(raw_points, timestamp)
        self._last_landmarks = hand_landmarks
        return {"points_px": filtered_points, "hand_scale": hand_scale, "landmarks": hand_landmarks}

    def draw_debug(self, frame) -> None:
        """Overlay the last detected hand landmarks for debugging."""
        if self._last_landmarks is None:
            return
        self._mp_draw.draw_landmarks(frame, self._last_landmarks, self._mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        self._hands.close()

