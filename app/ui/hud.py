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
        instructions = "Point to enter TRANSFORM | Open palm to enter CREATE"
    elif mode == "CREATE":
        instructions = "Pinch to create a shape | Open palm to exit"
    elif mode == "TRANSFORM":
        if has_selection:
            instructions = "Move hand to translate | Pinch to scale | Two fingers to rotate"
        else:
            instructions = "Point at a shape and pinch to select | Pointing gesture to exit"
    else:
        instructions = ""

    cv2.putText(frame, instructions, (10, h - 20), FONT, 0.7, COLOR_BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, instructions, (10, h - 20), FONT, 0.7, COLOR_WHITE, 1, cv2.LINE_AA)

