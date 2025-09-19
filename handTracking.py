import cv2
import mediapipe as mp

# Mediapipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        image = cap.read()

        # In RGB konvertieren (für MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hände erkennen
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Bild anzeigen
        cv2.imshow("Hand Gesture Prototype", image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC zum Beenden
            break

cap.release()
cv2.destroyAllWindows()
