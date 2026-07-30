import cv2
import mediapipe as mp
import requests

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

session = requests.Session()
ESP32_IP = "192.168.4.1"

def send_command(command):
    try:
        url = f"http://{ESP32_IP}/cmd?dir={command}"
        response = session.get(url, timeout=0.2)
        response.close()
        print("Sent:", command)
    except requests.exceptions.RequestException:
        print("ESP32 not reachable")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Gesture detection
def detect_gesture(landmarks):

    thumb_tip = landmarks[4]

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]

    index_open = index_tip.y < index_pip.y
    middle_open = middle_tip.y < middle_pip.y
    ring_open = ring_tip.y < ring_pip.y
    pinky_open = pinky_tip.y < pinky_pip.y

    fingers = [
        index_open,
        middle_open,
        ring_open,
        pinky_open
    ]

    count = fingers.count(True)

    if index_open and middle_open and not ring_open and not pinky_open:
        return "PEACE"

    if index_open and not middle_open and not ring_open and not pinky_open:
        return "POINT"

    if count == 4:
        return "PALM"

    if count == 0:
        return "FIST"

    if thumb_tip.y < landmarks[3].y and count == 0:
        return "THUMB UP"

    return "UNKNOWN"


# Camera
cap = cv2.VideoCapture(0)

print("GESTURE SYSTEM STARTED")
send_command("S")   # Stop the car when the program starts

previous_gestures = {}
previous_command = ""
hand_present = False

display_command = {
    "F": "FORWARD",
    "B": "REVERSE",
    "L": "LEFT",
    "R": "RIGHT",
    "S": "STOP",
    "NONE": "NONE",
    "NO HAND": "NO HAND"
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        left_gesture = None
        right_gesture = None

        for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness):

            label = handedness.classification[0].label

            lm_list = []

            for lm in hand_landmarks.landmark:
                lm_list.append(lm)

            gesture = detect_gesture(lm_list)

            # Store gesture for each hand
            if label == "Left":
                left_gesture = gesture
            else:
                right_gesture = gesture

            # Print only when gesture changes
            if previous_gestures.get(label) != gesture:

                print(label, ":", gesture)

                previous_gestures[label] = gesture

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Text near wrist
            h, w, _ = frame.shape

            wrist = hand_landmarks.landmark[0]

            x = int(wrist.x * w)
            y = int(wrist.y * h)

            cv2.putText(
                frame,
                label + ": " + gesture,
                (x - 60, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # ===========================
        # RC CAR COMMAND MAPPING
        # ===========================

        command = "NONE"

        if left_gesture == "FIST" and right_gesture == "FIST":
            command = "F"

        elif left_gesture == "FIST":
            command = "L"

        elif right_gesture == "FIST":
            command = "R"

        elif left_gesture == "PALM" and right_gesture == "PALM":
            command = "S"

        elif left_gesture == "POINT" and right_gesture == "POINT":
            command = "B"

        # Print only if command changes
        if command != previous_command:

            print("COMMAND:", display_command[command])

            send_command(command)

            previous_command = command
        # Choose colour
        color = (255, 255, 255)

        if command == "F":
            color = (0, 255, 0)          # Green

        elif command == "L":
            color = (255, 0, 0)          # Blue

        elif command == "R":
            color = (0, 255, 255)        # Yellow

        elif command == "S":
            color = (0, 0, 255)          # Red

        elif command == "B":
            color = (255, 0, 255)        # Purple
  
        
        # Draw command
        cv2.putText(
            frame,
            f"COMMAND: {display_command[command]}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
            )

        hand_present = True

    else:


        command = "NO HAND"
        if previous_command != "NO HAND":
            send_command("S")
            previous_command = "NO HAND"

        cv2.putText(
            frame,
            "COMMAND: NO HAND",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        if hand_present:

            print("NO HAND")

            previous_gestures.clear()

            previous_command = "NO HAND"

            hand_present = False

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

send_command("S")
session.close()
cap.release()
cv2.destroyAllWindows()