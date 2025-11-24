from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from app.input import features
from app.models.shapes import PolylineShape, Shape2D
from app.render import render2d
from app.state import modes

COLOR_SELECTED = (50, 255, 50)
DEFAULT_COLOR = (255, 100, 50)
STROKE_MIN_STEP = 5.0
STROKE_MIN_POINTS = 5
STROKE_EPSILON = 4.0


class Controller:
    """Owns shapes, selection, and mode transitions."""

    def __init__(self, screen_width: int = 1280, screen_height: int = 720) -> None:
        self.mode: str = modes.IDLE
        self.shapes: List[Shape2D] = [
            Shape2D(center=np.array([screen_width / 2, screen_height / 2]), size=150, color=DEFAULT_COLOR)
        ]
        self.selected_shape: Optional[Shape2D] = None

        # Debounce state
        self._last_gesture: Optional[str] = None
        self._gesture_frames: int = 0
        self._frame_index: int = 0
        self._mode_cooldown_frames = 30
        self._last_mode_switch_frame = -self._mode_cooldown_frames
        self._debounce_frames = 10

        # Transform baselines
        self.grab_offset: Optional[np.ndarray] = None
        self.base_size: Optional[float] = None
        self.base_pinch: Optional[float] = None
        self.base_shape_angle: Optional[float] = None
        self.base_finger_angle: Optional[float] = None

        # Drawing
        self.current_stroke: list[np.ndarray] = []
        self.is_drawing: bool = False

    def update(self, gestures: Optional[dict], points: Optional[Sequence[Sequence[float]]]) -> None:
        self._frame_index += 1
        if gestures is None or points is None:
            return

        self._maybe_switch_mode(gestures)

        if self.mode == modes.IDLE:
            self._deselect()
        elif self.mode == modes.CREATE:
            self._handle_create_mode(gestures, points)
        elif self.mode == modes.TRANSFORM:
            self._handle_transform_mode(gestures, points)
        elif self.mode == modes.DRAW:
            self._handle_draw_mode(gestures, points)

    def render(self, frame) -> None:
        for shape in self.shapes:
            shape.update()
            render2d.draw_shape(frame, shape, highlight=shape.is_selected, highlight_color=COLOR_SELECTED)
        if self.is_drawing and len(self.current_stroke) > 1:
            temp_poly = PolylineShape(points=self.current_stroke, color=(0, 200, 255), thickness=2)
            render2d.draw_shape(frame, temp_poly, highlight=False, highlight_color=COLOR_SELECTED)

    def _maybe_switch_mode(self, gestures: dict) -> None:
        persistent = None
        if gestures.get("is_open_palm"):
            persistent = "open_palm"
        elif gestures.get("is_pointing"):
            persistent = "pointing"
        elif gestures.get("is_draw_gesture"):
            persistent = "draw"

        if persistent != self._last_gesture:
            self._last_gesture = persistent
            self._gesture_frames = 0

        if persistent is None:
            return

        self._gesture_frames += 1
        if (
            self._gesture_frames >= self._debounce_frames
            and self._frame_index - self._last_mode_switch_frame >= self._mode_cooldown_frames
        ):
            if persistent == "open_palm":
                self._toggle_mode(modes.CREATE)
            elif persistent == "pointing":
                self._toggle_mode(modes.TRANSFORM)
            elif persistent == "draw":
                self._toggle_mode(modes.DRAW)
            self._last_mode_switch_frame = self._frame_index
            self._gesture_frames = 0

    def _toggle_mode(self, target_mode: str) -> None:
        if self.mode == target_mode:
            self.mode = modes.IDLE
        else:
            self.mode = target_mode
        if self.mode != modes.TRANSFORM:
            self._deselect()
        if self.mode != modes.DRAW:
            self._end_stroke()

    def _deselect(self) -> None:
        if self.selected_shape:
            self.selected_shape.is_selected = False
        self.selected_shape = None
        self.grab_offset = None
        self.base_size = None
        self.base_pinch = None
        self.base_shape_angle = None
        self.base_finger_angle = None

    def _handle_create_mode(self, gestures: dict, points: Sequence[Sequence[float]]) -> None:
        if not gestures.get("is_pinching"):
            return
        index_tip = points[8]
        for shape in self.shapes:
            if isinstance(shape, Shape2D) and features.distance(shape.center, index_tip) < shape.size:
                return
        self.shapes.append(Shape2D(center=np.array(index_tip), size=100, color=(0, 165, 255)))
        self.mode = modes.IDLE

    def _handle_transform_mode(self, gestures: dict, points: Sequence[Sequence[float]]) -> None:
        if self.selected_shape is None:
            if not gestures.get("is_pinching"):
                return
            index_tip = points[8]
            for shape in reversed(self.shapes):
                if isinstance(shape, Shape2D) and shape.contains_point(index_tip):
                    self._select_shape(shape, points, gestures)
                    break
        else:
            if not gestures.get("is_pinching") and not gestures.get("is_two_finger_point"):
                self._deselect()
                return
            self._apply_transform(gestures, points)

    def _select_shape(self, shape: Shape2D, points: Sequence[Sequence[float]], gestures: dict) -> None:
        for s in self.shapes:
            s.is_selected = False
        shape.is_selected = True
        self.selected_shape = shape
        anchor = np.asarray(points[9], dtype=float)
        self.grab_offset = shape.center - anchor
        self.base_size = shape.size
        self.base_pinch = gestures.get("pinch_ratio")
        self.base_shape_angle = shape.angle
        self.base_finger_angle = features.angle_degrees(points[8], points[12])

    def _apply_transform(self, gestures: dict, points: Sequence[Sequence[float]]) -> None:
        shape = self.selected_shape
        if shape is None or not isinstance(shape, Shape2D):
            return

        anchor = np.asarray(points[9], dtype=float)  # middle finger MCP for stability
        target_center = anchor + (self.grab_offset if self.grab_offset is not None else 0)
        shape.target_center = target_center

        pinch_ratio = gestures.get("pinch_ratio", 0.0)
        if self.base_pinch and pinch_ratio > 1e-6:
            scale_factor = self.base_pinch / pinch_ratio
            shape.target_size = features.clamp(self.base_size * scale_factor, 20.0, 600.0)

        if gestures.get("is_two_finger_point"):
            finger_angle = features.angle_degrees(points[8], points[12])
            if self.base_finger_angle is None:
                self.base_finger_angle = finger_angle
                self.base_shape_angle = shape.angle
            angle_delta = finger_angle - self.base_finger_angle
            shape.target_angle = self.base_shape_angle + angle_delta
        else:
            self.base_finger_angle = None
            self.base_shape_angle = shape.angle

    def _handle_draw_mode(self, gestures: dict, points: Sequence[Sequence[float]]) -> None:
        draw_active = bool(gestures.get("is_draw_gesture"))
        index_tip = np.asarray(points[8], dtype=float)

        if draw_active and not self.is_drawing:
            self.is_drawing = True
            self.current_stroke = [index_tip]
        elif draw_active and self.is_drawing:
            if self.current_stroke:
                last_pt = self.current_stroke[-1]
                if np.linalg.norm(index_tip - last_pt) >= STROKE_MIN_STEP:
                    self.current_stroke.append(index_tip)
        elif self.is_drawing and not draw_active:
            self._end_stroke()

    def _end_stroke(self) -> None:
        if not self.current_stroke:
            self.is_drawing = False
            return
        points = self.current_stroke
        self.current_stroke = []
        self.is_drawing = False

        if len(points) < STROKE_MIN_POINTS:
            self.mode = modes.IDLE
            return

        simplified = features.rdp(points, epsilon=STROKE_EPSILON)
        polyline = PolylineShape(points=[np.asarray(p, dtype=float) for p in simplified], color=(0, 0, 255), thickness=3)
        self.shapes.append(polyline)
        self.mode = modes.IDLE
