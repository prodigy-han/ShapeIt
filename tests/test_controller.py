import numpy as np
import pytest

from app.state.controller import Controller
from app.state import modes


def test_transform_baseline_updates_targets():
    controller = Controller()
    shape = controller.shapes[0]
    controller.selected_shape = shape
    controller.grab_offset = np.array([10.0, -5.0])
    controller.base_size = 100.0
    controller.base_pinch = 0.5
    controller.base_shape_angle = 10.0
    controller.base_finger_angle = 0.0
    controller.is_grabbing = True

    points = [(0.0, 0.0)] * 21
    points[9] = (100.0, 100.0)  # anchor
    points[5] = (110.0, 100.0)  # index MCP to define rotation vector
    points[8] = (110.0, 0.0)
    points[12] = (110.0, 0.0)  # unused now

    gestures = {"pinch_ratio": 0.25, "is_two_finger_point": True}

    controller._apply_transform(gestures, points)

    assert np.allclose(shape.target_center, np.array([110.0, 95.0]))
    assert shape.target_size == pytest.approx(50.0)
    assert shape.target_angle == pytest.approx(-80.0)


def test_exit_gesture_clears_selection_and_mode():
    controller = Controller()
    controller.mode = modes.TRANSFORM
    controller.selected_shape = controller.shapes[0]
    controller.is_grabbing = True
    controller.current_stroke = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    controller.is_drawing = True

    controller._exit_to_idle()

    assert controller.mode == modes.IDLE
    assert controller.selected_shape is None
    assert controller.is_grabbing is False
    assert controller.is_drawing is False
    assert controller.current_stroke == []
    assert controller._mode_switch_cooldown > 0


def test_exit_from_create_does_not_spawn_shape():
    controller = Controller()
    controller.mode = modes.CREATE
    start_len = len(controller.shapes)
    gestures = {"is_exit_gesture": True}
    points = [(0.0, 0.0)] * 21
    controller.update(gestures, points)
    assert controller.mode == modes.IDLE
    assert len(controller.shapes) == start_len


def test_exit_from_transform_and_draw():
    controller = Controller()
    points = [(0.0, 0.0)] * 21

    controller.mode = modes.TRANSFORM
    controller.selected_shape = controller.shapes[0]
    controller.is_grabbing = True
    controller.update({"is_exit_gesture": True}, points)
    assert controller.mode == modes.IDLE
    assert controller.selected_shape is None

    controller.mode = modes.DRAW
    controller.current_stroke = [np.array([0.0, 0.0])]
    controller.is_drawing = True
    controller.update({"is_exit_gesture": True}, points)
    assert controller.mode == modes.IDLE
    assert controller.is_drawing is False


def test_selection_persists_with_open_pinch():
    controller = Controller()
    controller.mode = modes.TRANSFORM
    shape = controller.shapes[0]
    cx, cy = shape.center

    points = [(0.0, 0.0)] * 21
    points[5] = (cx - 10.0, cy)
    points[8] = (cx, cy)
    points[9] = (cx, cy)

    gestures_select = {"is_pinching": True, "pinch_ratio": 0.3}
    controller._handle_transform_mode(gestures_select, points)
    assert controller.selected_shape is shape

    gestures_open = {"is_pinching": False, "pinch_ratio": 1.1}
    controller._handle_transform_mode(gestures_open, points)

    assert controller.selected_shape is shape
    assert controller.is_grabbing is True


def test_exit_cooldown_blocks_immediate_mode_switch():
    controller = Controller()
    controller.mode = modes.TRANSFORM
    controller._exit_to_idle()
    assert controller.mode == modes.IDLE
    controller._maybe_switch_mode({"is_open_palm": True, "is_pointing": False, "is_three_finger": False, "is_draw_active": False})
    assert controller.mode == modes.IDLE  # still cooling down
    controller._mode_switch_cooldown = 0
    controller._maybe_switch_mode({"is_open_palm": True, "is_pointing": False, "is_three_finger": False, "is_draw_active": False})
    assert controller.mode == modes.CREATE


def test_mode_switches_from_idle():
    controller = Controller()
    shape = controller.shapes[0]
    controller.selected_shape = shape

    controller._maybe_switch_mode({"is_open_palm": True, "is_pointing": False, "is_three_finger": False, "is_draw_active": False})
    assert controller.mode == modes.CREATE
    assert controller.selected_shape is None

    controller._maybe_switch_mode({"is_open_palm": False, "is_pointing": True, "is_three_finger": False, "is_draw_active": False})
    assert controller.mode == modes.TRANSFORM

    controller.mode = modes.IDLE
    controller._maybe_switch_mode({"is_open_palm": False, "is_pointing": False, "is_three_finger": True, "is_draw_active": True})
    assert controller.mode == modes.DRAW


def test_draw_mode_ignores_open_palm():
    controller = Controller()
    controller.mode = modes.DRAW
    controller._maybe_switch_mode({"is_open_palm": True, "is_pointing": False, "is_three_finger": False, "is_draw_active": False})
    assert controller.mode == modes.DRAW
