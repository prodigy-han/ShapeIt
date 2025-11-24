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


def test_draw_and_exit_debounce():
    engine = GestureInterpreter(pinch_start=0.35, pinch_end=0.45)

    def three_finger_points():
        pts = [(0.0, 60.0)] * 21
        # index
        pts[6] = (0.0, 60.0)
        pts[8] = (0.0, 40.0)
        # middle
        pts[10] = (0.0, 60.0)
        pts[12] = (0.0, 40.0)
        # ring
        pts[14] = (0.0, 60.0)
        pts[16] = (0.0, 40.0)
        # pinky down
        pts[18] = (0.0, 60.0)
        pts[20] = (0.0, 80.0)
        # thumb slightly open
        pts[3] = (-5.0, 50.0)
        pts[4] = (5.0, 50.0)
        return pts

    res1 = engine.compute(three_finger_points(), hand_scale=100.0)
    res2 = engine.compute(three_finger_points(), hand_scale=100.0)
    res3 = engine.compute(three_finger_points(), hand_scale=100.0)
    assert res3["is_three_finger"] is True
    assert res3["is_draw_active"] is True
    assert res1["is_draw_active"] is False

    def fist_points():
        pts = [(0.0, 60.0)] * 21
        pts[3] = (0.0, 60.0)
        pts[4] = (0.0, 60.0)
        return pts

    e1 = engine.compute(fist_points(), hand_scale=100.0)
    e2 = engine.compute(fist_points(), hand_scale=100.0)
    e3 = engine.compute(fist_points(), hand_scale=100.0)
    assert e3["is_exit_gesture"] is True
    assert e1["is_exit_gesture"] is False


def test_thumb_up_flag():
    engine = GestureInterpreter()
    pts = [(0.0, 60.0)] * 21
    # thumb extended
    pts[3] = (-5.0, 60.0)
    pts[4] = (20.0, 60.0)
    # other fingers down
    pts[6] = (0.0, 70.0)
    pts[8] = (0.0, 80.0)
    pts[10] = (0.0, 70.0)
    pts[12] = (0.0, 80.0)
    pts[14] = (0.0, 70.0)
    pts[16] = (0.0, 80.0)
    pts[18] = (0.0, 70.0)
    pts[20] = (0.0, 80.0)
    res = engine.compute(pts, hand_scale=100.0)
    assert res["is_thumb_up"] is True
