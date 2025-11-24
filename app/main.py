import time

import cv2

from app.gestures.rules import GestureInterpreter
from app.input.hand_tracking import HandTracker
from app.state.controller import Controller
from app.ui.hud import draw_hud

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    hand_tracker = HandTracker()
    gesture_engine = GestureInterpreter()
    controller = Controller(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        timestamp = time.time()

        hand_result = hand_tracker.get_landmarks(frame, timestamp=timestamp)
        if hand_result:
            points = hand_result["points_px"]
            gestures = gesture_engine.compute(points, hand_result["hand_scale"])
            controller.update(gestures, points)
            hand_tracker.draw_debug(frame)
        else:
            controller.update(None, None)

        controller.render(frame)
        draw_hud(frame, controller.mode, controller.selected_shape is not None)

        cv2.imshow("Hand-Gesture Shape Sculptor", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()


if __name__ == "__main__":
    main()

