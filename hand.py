import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- Constants and Configuration ---
# Display
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Smoothing
SMOOTHING_FACTOR = 0.5  # Lower value = more smoothing, but more lag

# Colors (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 100, 50)
COLOR_GREEN = (50, 255, 50)
COLOR_RED = (50, 50, 255)
COLOR_PURPLE = (255, 50, 255)
COLOR_ORANGE = (0, 165, 255)

# Gesture Debouncing
GESTURE_DEBOUNCE_FRAMES = 10  # Number of frames a gesture must be held to trigger a mode switch

class Shape:
    """A class to represent a manipulable shape."""
    def __init__(self, center, size, color=COLOR_BLUE):
        self.center = np.array(center, dtype=float)
        self.target_center = np.array(center, dtype=float)

        self.size = float(size)
        self.target_size = float(size)

        self.angle = 0.0
        self.target_angle = 0.0

        self.color = color
        self.is_selected = False

    def contains_point(self, point):
        """Check if a point is inside the shape's bounding box."""
        half_size = self.size / 2
        return (self.center[0] - half_size < point[0] < self.center[0] + half_size and
                self.center[1] - half_size < point[1] < self.center[1] + half_size)

    def update(self):
        """Apply smoothing to shape properties."""
        self.center += (self.target_center - self.center) * SMOOTHING_FACTOR
        self.size += (self.target_size - self.size) * SMOOTHING_FACTOR
        
        # Smooth angle, handling wraparound from 360 to 0 degrees
        angle_diff = self.target_angle - self.angle
        if angle_diff > 180: angle_diff -= 360
        if angle_diff < -180: angle_diff += 360
        self.angle += angle_diff * SMOOTHING_FACTOR
        self.angle %= 360 # Keep angle within [0, 360)

    def draw(self, frame):
        """Draw the rotated rectangle on the frame."""
        color = COLOR_GREEN if self.is_selected else self.color
        
        # Calculate vertices of the rotated rectangle
        x, y = self.center
        s = int(self.size / 2)
        angle_rad = np.deg2rad(self.angle)
        
        points = np.array([
            [-s, -s], [s, -s], [s, s], [-s, s]
        ])
        
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        
        rotated_points = np.dot(points, rotation_matrix.T) + self.center
        pts = rotated_points.astype(int).reshape((-1, 1, 2))
        
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
        
        # Draw a small circle at the center
        cv2.circle(frame, tuple(self.center.astype(int)), 5, color, -1)


class HandGestureController:
    """Manages hand tracking, gesture recognition, and shape manipulation."""

    # --- Modes ---
    MODE_IDLE = "IDLE"
    MODE_TRANSFORM = "TRANSFORM"
    MODE_CREATE = "CREATE"

    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

        # Application state
        self.shapes = [Shape(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), size=150)]
        self.mode = self.MODE_IDLE
        self.selected_shape = None
        self.active_hand_landmarks = None

        # Gesture state
        self.gesture_start_time = None
        self.last_gesture = None
        self.pinch_base_dist = None
        self.rotation_base_angle = None
        
        self.last_mode_switch_time = 0

    def _get_landmarks(self, frame):
        """Process a frame to find hand landmarks."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            self.active_hand_landmarks = landmarks
            
            # Convert normalized landmarks to pixel coordinates
            return [(int(p.x * w), int(p.y * h)) for p in landmarks.landmark]
        
        self.active_hand_landmarks = None
        return None
    
    @staticmethod
    def _dist(p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _detect_gestures(self, points):
        """Detect gestures based on finger positions."""
        # Simple finger-up check (based on y-coordinates relative to knuckles)
        index_up = points[8][1] < points[6][1]
        middle_up = points[12][1] < points[10][1]
        ring_up = points[16][1] < points[14][1]
        pinky_up = points[20][1] < points[18][1]
        thumb_up = points[4][0] > points[3][0] # Simple check for thumb being 'out'

        # Pinch gesture
        pinch_dist = self._dist(points[4], points[8])
        is_pinching = pinch_dist < 40

        # Two-finger point (for rotation)
        is_two_finger_point = index_up and middle_up and not ring_up and not pinky_up

        # Open Palm gesture
        is_open_palm = index_up and middle_up and ring_up and pinky_up

        # Pointing gesture
        is_pointing = index_up and not middle_up and not ring_up and not pinky_up

        # Return a dictionary of detected gestures
        return {
            "pinch_dist": pinch_dist,
            "is_pinching": is_pinching,
            "is_two_finger_point": is_two_finger_point,
            "is_open_palm": is_open_palm,
            "is_pointing": is_pointing
        }

    def _update_state(self, gestures, points):
        """Update the application state based on gestures and mode."""
        current_time = time.time()
        
        # --- Mode Switching Logic with Debouncing ---
        current_persistent_gesture = None
        if gestures["is_open_palm"]:
            current_persistent_gesture = "open_palm"
        elif gestures["is_pointing"]:
            current_persistent_gesture = "pointing"
            
        if current_persistent_gesture != self.last_gesture:
            self.last_gesture = current_persistent_gesture
            self.gesture_start_time = current_time
        
        # Check if gesture has been held long enough
        if self.gesture_start_time and (current_time - self.gesture_start_time > GESTURE_DEBOUNCE_FRAMES / 30.0):
             # Check for cooldown to prevent rapid switching
            if current_time - self.last_mode_switch_time > 1.0:
                if self.last_gesture == "open_palm":
                    self.mode = self.MODE_CREATE if self.mode != self.MODE_CREATE else self.MODE_IDLE
                    print(f"Switched mode to: {self.mode}")
                    self.gesture_start_time = None # Reset to prevent re-triggering
                    self.last_mode_switch_time = current_time
                    self.selected_shape = None # Deselect on mode change
                    for shape in self.shapes: shape.is_selected = False
                elif self.last_gesture == "pointing":
                    self.mode = self.MODE_TRANSFORM if self.mode != self.MODE_TRANSFORM else self.MODE_IDLE
                    print(f"Switched mode to: {self.mode}")
                    self.gesture_start_time = None
                    self.last_mode_switch_time = current_time


        # --- Mode-Specific Actions ---
        index_tip = points[8]

        # IDLE Mode: Do nothing
        if self.mode == self.MODE_IDLE:
            self.selected_shape = None
            for shape in self.shapes:
                shape.is_selected = False

        # CREATE Mode: Pinch to create a new shape
        elif self.mode == self.MODE_CREATE:
            if gestures["is_pinching"]:
                 # Check if a shape already exists nearby to avoid spamming
                can_create = all(self._dist(index_tip, s.center) > s.size for s in self.shapes)
                if can_create:
                    new_shape = Shape(center=index_tip, size=100, color=COLOR_ORANGE)
                    self.shapes.append(new_shape)
                    print("Created new shape")
                    self.mode = self.MODE_IDLE # Return to IDLE after creation
            
        # TRANSFORM Mode: Select and manipulate shapes
        elif self.mode == self.MODE_TRANSFORM:
            if self.selected_shape is None:
                # Selection logic
                if gestures["is_pinching"]:
                    for shape in reversed(self.shapes): # Check from top-most shape
                        if shape.contains_point(index_tip):
                            self.selected_shape = shape
                            shape.is_selected = True
                            print("Shape selected")
                            
                            # Calibrate for scaling and rotation
                            self.pinch_base_dist = gestures["pinch_dist"]
                            p1 = points[8] # Index tip
                            p2 = points[12] # Middle tip
                            self.rotation_base_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
                            break
            else:
                # Deselection
                if not gestures["is_pinching"] and not gestures["is_two_finger_point"]:
                    print("Shape deselected")
                    self.selected_shape.is_selected = False
                    self.selected_shape = None
                    self.pinch_base_dist = None
                    self.rotation_base_angle = None
                    return
                
                # --- Transformations ---
                # 1. Translate
                self.selected_shape.target_center = np.array(points[9]) # Use middle finger MCP for stability

                # 2. Scale
                if self.pinch_base_dist:
                    scale_factor = self.pinch_base_dist / max(1, gestures["pinch_dist"])
                    self.selected_shape.target_size = np.clip(100 * scale_factor, 20, 500)

                # 3. Rotate
                if gestures["is_two_finger_point"]:
                    p1 = points[8]
                    p2 = points[12]
                    current_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
                    
                    if self.rotation_base_angle is not None:
                        angle_delta = current_angle - self.rotation_base_angle
                        self.selected_shape.target_angle = self.selected_shape.angle + angle_delta
                else:
                    # If not rotating, reset the base angle to prevent jumps
                    self.rotation_base_angle = None


    def _draw_hud(self, frame):
        """Draw the Heads-Up Display with mode and instructions."""
        # Mode display
        mode_text = f"MODE: {self.mode}"
        cv2.putText(frame, mode_text, (10, 40), FONT, 1, COLOR_BLACK, 5, cv2.LINE_AA)
        cv2.putText(frame, mode_text, (10, 40), FONT, 1, COLOR_WHITE, 2, cv2.LINE_AA)

        # Instructions
        if self.mode == self.MODE_IDLE:
            instructions = "Point to enter TRANSFORM | Open Palm to enter CREATE"
        elif self.mode == self.MODE_CREATE:
            instructions = "Pinch to create a shape | Open Palm to exit"
        elif self.mode == self.MODE_TRANSFORM:
            if self.selected_shape is None:
                instructions = "Point at a shape and Pinch to select | Pointing gesture to exit"
            else:
                instructions = "Move hand to Translate | Pinch to Scale | Use two fingers to Rotate"
        else:
            instructions = ""
        
        cv2.putText(frame, instructions, (10, SCREEN_HEIGHT - 20), FONT, 0.7, COLOR_BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, instructions, (10, SCREEN_HEIGHT - 20), FONT, 0.7, COLOR_WHITE, 1, cv2.LINE_AA)

    def run(self):
        """Main application loop."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)

            # Process hand
            points = self._get_landmarks(frame)

            if points:
                # Detect gestures
                gestures = self._detect_gestures(points)
                
                # Update state
                self._update_state(gestures, points)

                # Draw hand landmarks for debugging
                self.mp_draw.draw_landmarks(
                    frame, self.active_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw circles on keypoints for feedback
                cv2.circle(frame, points[4], 10, COLOR_PURPLE, -1) # Thumb
                cv2.circle(frame, points[8], 10, COLOR_PURPLE, -1) # Index
                cv2.circle(frame, points[12], 10, COLOR_ORANGE, -1) # Middle
            
            # Update and draw all shapes
            for shape in self.shapes:
                shape.update()
                shape.draw(frame)

            # Draw HUD
            self._draw_hud(frame)

            cv2.imshow('Hand-Gesture Shape Sculptor', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()
