import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model + label encoder
model = load_model("sign_alphabet_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction_text = "No hand detected"

    if result.multi_hand_landmarks:
        landmarks = []

        # Collect features for each detected hand
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # If only 1 hand → pad with zeros (63 values)
        if len(result.multi_hand_landmarks) == 1:
            landmarks.extend([0.0] * (21 * 3))

        # If >2 hands (rare) → trim extra
        if len(landmarks) > 126:
            landmarks = landmarks[:126]

        # Ensure always exactly 126 features
        if len(landmarks) == 126:
            X = np.array(landmarks).reshape(1, -1)

            # Predict
            preds = model.predict(X)
            pred_class = np.argmax(preds)
            prediction_text = le.inverse_transform([pred_class])[0]

        # Draw landmarks on screen
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Prediction: {prediction_text}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Alphabet Recognition - Both Hands", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
