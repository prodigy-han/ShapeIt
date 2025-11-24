import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def draw_hud(frame, mode: str, has_selection: bool) -> None:
    h, w, _ = frame.shape
    mode_text = f"MODE: {mode}"
    cv2.putText(frame, mode_text, (10, 40), FONT, 1, COLOR_BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, mode_text, (10, 40), FONT, 1, COLOR_WHITE, 2, cv2.LINE_AA)

    if mode == "IDLE":
        instructions = "Point: TRANSFORM | Open palm: CREATE | Three fingers: DRAW | Fist: reset"
    elif mode == "CREATE":
        instructions = "Thumbs up to create a shape | Fist: exit to IDLE"
    elif mode == "TRANSFORM":
        instructions = (
            "Move: move shape | Pinch: scale | Twist index: rotate | Fist: exit to IDLE"
            if has_selection
            else "Point at a shape and pinch to select | Fist: exit to IDLE"
        )
    elif mode == "DRAW":
        instructions = "Hold three fingers to draw | Release to finish | Fist: cancel and return to IDLE"
    else:
        instructions = ""

    cv2.putText(frame, instructions, (10, h - 20), FONT, 0.7, COLOR_BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, instructions, (10, h - 20), FONT, 0.7, COLOR_WHITE, 1, cv2.LINE_AA)
