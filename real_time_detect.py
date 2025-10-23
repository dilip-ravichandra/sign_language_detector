import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load model and label encoder
with open('sign_language_model.h5', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            input_data = np.array(landmark_list).reshape(1, -1)

            # Predict
            prediction = model.predict(input_data)
            label = le.inverse_transform(prediction)[0]

            # Draw landmarks and label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)



    cv2.imshow('Sign Detection', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
