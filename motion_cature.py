import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define labels
labels = ["my_name_is_dilip"]
data_path = "Motion_Data"
os.makedirs(data_path, exist_ok=True)

# Capture from webcam
cap = cv2.VideoCapture(0)
sequence_length = 30  # number of frames per motion sequence

print("Press 'm' to record 'my name is Dilip' motion (30 frames). Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Motion Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # Start recording motion sequence
    if key == ord('m'):
        print("Recording motion sequence...")
        sequence = []
        for _ in range(sequence_length):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            landmarks = []
            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks = [0] * 63  # fallback if no hand detected (21 points × 3)

            sequence.append(landmarks)

            # Draw for visualization
            if result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Motion Capture", frame)
            cv2.waitKey(30)

        # Save motion sequence
        seq_array = np.array(sequence)
        file_path = os.path.join(data_path, f"my_name_is_dilip_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy")
        np.save(file_path, seq_array)
        print(f"Saved motion sequence at {file_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
