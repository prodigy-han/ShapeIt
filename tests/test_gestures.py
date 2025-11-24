from app.gestures.rules import GestureInterpreter


def _make_points(thumb_tip, index_tip):
    # Create a minimal set of landmarks for gesture checks.
    points = [(0.0, 50.0)] * 21
    points[3] = (thumb_tip[0] - 5.0, thumb_tip[1])  # thumb joint for simple open check
    points[4] = thumb_tip
    points[6] = (index_tip[0], index_tip[1] + 10.0)  # index PIP below tip => finger up
    points[8] = index_tip
    points[10] = (0.0, 60.0)  # middle pip
    points[12] = (0.0, 70.0)  # middle tip
    points[14] = (0.0, 60.0)
    points[16] = (0.0, 70.0)
    points[18] = (0.0, 60.0)
    points[20] = (0.0, 70.0)
    return points


def test_pinch_hysteresis_toggles():
    engine = GestureInterpreter(pinch_start=0.35, pinch_end=0.45)
    close_points = _make_points((0.0, 0.0), (30.0, 0.0))  # ratio 0.3 with scale 100
    gestures = engine.compute(close_points, hand_scale=100.0)
    assert gestures["pinch_ratio"] == 0.3
    assert gestures["is_pinching"] is True

    far_points = _make_points((0.0, 0.0), (60.0, 0.0))  # ratio 0.6 with scale 100
    gestures = engine.compute(far_points, hand_scale=100.0)
    assert gestures["pinch_ratio"] == 0.6
    assert gestures["is_pinching"] is False

