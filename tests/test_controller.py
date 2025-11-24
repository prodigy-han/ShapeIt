import numpy as np
import pytest

from app.state.controller import Controller


def test_transform_baseline_updates_targets():
    controller = Controller()
    shape = controller.shapes[0]
    controller.selected_shape = shape
    controller.grab_offset = np.array([10.0, -5.0])
    controller.base_size = 100.0
    controller.base_pinch = 0.5
    controller.base_shape_angle = 10.0
    controller.base_finger_angle = 0.0

    points = [(0.0, 0.0)] * 21
    points[9] = (100.0, 100.0)  # anchor
    points[8] = (110.0, 100.0)
    points[12] = (110.0, 0.0)  # finger angle -90 degrees

    gestures = {"pinch_ratio": 0.25, "is_two_finger_point": True}

    controller._apply_transform(gestures, points)

    assert np.allclose(shape.target_center, np.array([110.0, 95.0]))
    assert shape.target_size == pytest.approx(200.0)
    assert shape.target_angle == pytest.approx(-80.0)

