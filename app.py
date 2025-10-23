from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and label encoder from alphabet_data folder
model = load_model("alphabet_data/sign_alphabet_model.h5")
with open("alphabet_data/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

wait_time = 5
countdown_started = False
start_time = None
sentence = ""

def generate_frames():
    global countdown_started, start_time, sentence
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        current_time = time.time()

        if result.multi_hand_landmarks:
            if not countdown_started:
                countdown_started = True
                start_time = current_time

            elapsed = current_time - start_time
            remaining_time = int(wait_time - elapsed)

            if remaining_time > 0:
                cv2.putText(frame, f"Predicting in: {remaining_time}s", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                landmarks = []
                for hand_landmarks in result.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                if len(result.multi_hand_landmarks) == 1:
                    landmarks.extend([0.0] * (21 * 3))

                if len(landmarks) >= 126:
                    landmarks = landmarks[:126]
                    X = np.array(landmarks).reshape(1, -1)
                    preds = model.predict(X)
                    pred_class = np.argmax(preds)
                    predicted_letter = le.inverse_transform([pred_class])[0]
                    sentence += predicted_letter

                start_time = current_time

        else:
            countdown_started = False
            start_time = None

        cv2.putText(frame, sentence, (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
