import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Labels A–Z
labels = [chr(i) for i in range(65, 91)]  # A-Z
data_path = "Alphabet_Data"
os.makedirs(data_path, exist_ok=True)

# Create folders if not already present
for label in labels:
    os.makedirs(os.path.join(data_path, label), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press A–Z to capture one sample | Press '1' to quit")

landmarks_per_hand = 21 * 3  # 21 landmarks, each (x, y, z)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw detected hands
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, "Press A-Z to capture | Press 1 to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Data Collection (A–Z)", frame)

    key = cv2.waitKey(1) & 0xFF

    # When alphabet key (A–Z) pressed
    if ord('A') <= key <= ord('Z'):
        current_label = chr(key)
        landmarks = np.zeros(landmarks_per_hand * 2)  # space for both hands

        if result.multi_hand_landmarks:
            # Fill landmarks from available hands
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                if idx >= 2:
                    break
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.extend([lm.x, lm.y, lm.z])
                start = idx * landmarks_per_hand
                landmarks[start:start + landmarks_per_hand] = lm_list

            # Save the sample (one or both hands)
            npy_path = os.path.join(
                data_path, current_label,
                f'{current_label}_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.npy'
            )
            np.save(npy_path, landmarks)
            print(f"✅ Saved sample for {current_label}: {npy_path}")
        else:
            print("⚠️ No hands detected — try again!")

    # Quit on number key
    elif key in [ord(str(i)) for i in range(0, 10)]:
        print("👋 Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
