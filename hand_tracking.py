import cv2
import mediapipe as mp

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands object (detect max 2 hands)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,   # 👈 enable both hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip image for natural selfie-view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
            )

            # Show whether it's left or right hand
            label = handedness.classification[0].label  # 'Left' or 'Right'
            cv2.putText(frame, label, 
                        (int(hand_landmarks.landmark[0].x * frame.shape[1]),
                         int(hand_landmarks.landmark[0].y * frame.shape[0]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Two-Hand Tracking", frame)

    # Exit with ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
