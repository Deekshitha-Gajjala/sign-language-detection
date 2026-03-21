import cv2
import time
from hand_detector import HandDetector
from gesture_logic import detect_gesture
from text_to_speech import speak

detector = HandDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(3, 640)
cap.set(4, 480)

time.sleep(2)

last_spoken_time = 0
interval = 1.5   # speak every 1.5 sec

while True:
    success, frame = cap.read()

    if not success or frame is None:
        print("Camera not working")
        continue

    frame = cv2.flip(frame, 1)

    frame, landmarks = detector.detect_hands(frame)

    gesture = detect_gesture(landmarks)

    # Show number on screen
    if gesture:
        cv2.putText(frame, f"Fingers: {gesture}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    current_time = time.time()

    # 🔥 Speak number continuously
    if gesture and (current_time - last_spoken_time > interval):
        speak(gesture)
        last_spoken_time = current_time

    cv2.imshow("Sign App", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()