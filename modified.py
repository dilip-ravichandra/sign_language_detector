import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3  # 🔹 added for voice output
from tensorflow.keras.models import load_model

# Load trained model and label encoder
model = load_model("sign_alphabet_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 🔹 Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)    # speed of speech
engine.setProperty('volume', 1.0)  # max volume

wait_time = 3           # seconds to wait before each prediction
countdown_started = False
start_time = None
sentence = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        # Start countdown only once when hand is first seen
        if not countdown_started:
            countdown_started = True
            start_time = current_time

        elapsed = current_time - start_time
        remaining_time = int(wait_time - elapsed)

        if remaining_time > 0:
            cv2.putText(frame, f"Predicting in: {remaining_time}s", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # After 5 seconds → make prediction
            landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # Pad in case only one hand
            if len(result.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * (21 * 3))

            if len(landmarks) >= 126:
                landmarks = landmarks[:126]
                X = np.array(landmarks).reshape(1, -1)
                preds = model.predict(X)
                pred_class = np.argmax(preds)
                predicted_letter = le.inverse_transform([pred_class])[0]
                sentence += predicted_letter

            # Reset countdown for next prediction
            start_time = current_time

    else:
        # If no hand detected, reset countdown
        countdown_started = False
        start_time = None

    # 🔹 Key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):        # spacebar to add space
        sentence += " "
    elif key == 13:            # Enter key → speak the sentence
        if sentence.strip() != "":
            print(f"Speaking: {sentence}")
            engine.say(sentence)
            engine.runAndWait()
    elif key == ord('q'):      # quit
        break

    # Show subtitles
    cv2.putText(frame, sentence, (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Subtitles", frame)

cap.release()
cv2.destroyAllWindows()
