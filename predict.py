import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time

# Load model
model = tf.keras.models.load_model("sign_model.h5")

# Labels (make sure this matches your dataset folder names)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Text-to-speech
engine = pyttsx3.init()

last_time = 0
last_label = ""

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # Add margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop hand
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                # Resize to model input
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = hand_img / 255.0
                hand_img = np.reshape(hand_img, (1, 64, 64, 3))

                # Predict
                prediction = model.predict(hand_img, verbose=0)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]

                label = labels[class_id]

                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Show text
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                # Speak (no spam)
                if label != last_label and time.time() - last_time > 2:
                    print(f"Speaking: {label}")
                    engine.say(label)
                    engine.runAndWait()
                    last_time = time.time()
                    last_label = label

    # Show frame
    cv2.imshow("Sign Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()