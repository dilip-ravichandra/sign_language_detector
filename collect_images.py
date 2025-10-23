import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define labels
labels = ["hello", "yes", "no", "thankyou", "iloveyou"]
label_index = {label: idx for idx, label in enumerate(labels)}

# Set data path
data_path = 'MP_Data'
os.makedirs(data_path, exist_ok=True)

# Create folders if they don't exist
for label in labels:
    os.makedirs(os.path.join(data_path, label), exist_ok=True)

# Count existing samples to continue where left off
sample_count = {label: len(os.listdir(os.path.join(data_path, label))) for label in labels}

# Collect data
cap = cv2.VideoCapture(0)
print("Press corresponding key to capture ONE sample:")
print("h: hello | y: yes | n: no | i: iloveyou | t: thankyou | q: quit")

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

    # Display counts on screen
    y_offset = 30
    for lbl in labels:
        cv2.putText(frame, f'{lbl}: {sample_count[lbl]}/30', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow('Collect Landmarks', frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture on key press only
    if key == ord('h') and sample_count["hello"] < 30:
        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            npy_path = os.path.join(data_path, "hello",
                                    f'hello_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.npy')
            np.save(npy_path, np.array(landmarks))
            sample_count["hello"] += 1
            print(f"Captured hello {sample_count['hello']}/30")

    elif key == ord('y') and sample_count["yes"] < 30:
        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            npy_path = os.path.join(data_path, "yes",
                                    f'yes_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.npy')
            np.save(npy_path, np.array(landmarks))
            sample_count["yes"] += 1
            print(f"Captured yes {sample_count['yes']}/30")

    elif key == ord('n') and sample_count["no"] < 30:
        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            npy_path = os.path.join(data_path, "no",
                                    f'no_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.npy')
            np.save(npy_path, np.array(landmarks))
            sample_count["no"] += 1
            print(f"Captured no {sample_count['no']}/30")

    elif key == ord('i') and sample_count["iloveyou"] < 30:
        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            npy_path = os.path.join(data_path, "iloveyou",
                                    f'iloveyou_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.npy')
            np.save(npy_path, np.array(landmarks))
            sample_count["iloveyou"] += 1
            print(f"Captured iloveyou {sample_count['iloveyou']}/30")

    elif key == ord('t') and sample_count["thankyou"] < 30:
        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            npy_path = os.path.join(data_path, "thankyou",
                                    f'thankyou_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.npy')
            np.save(npy_path, np.array(landmarks))
            sample_count["thankyou"] += 1
            print(f"Captured thankyou {sample_count['thankyou']}/30")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
