import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Modell laden
model = tf.keras.models.load_model("silenceai\Training\models\gesture_model.h5")
CLASSES = ["Zwei","Vier"]  # passe an deine Klassen an

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# FPS Berechnung
prev_time = 0

def extract_keypoints(hand_landmarks):
    """Keypoints in 1D-Vektor umwandeln"""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# Kamera starten
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_text = "Keine Hand erkannt"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Keypoints extrahieren
                features = extract_keypoints(hand_landmarks).reshape(1, -1)

                # Vorhersage mit MLP
                probs = model.predict(features, verbose=0)[0]
                pred_class = np.argmax(probs)
                gesture_text = f"Buchstabe: {CLASSES[pred_class]}"

        # FPS berechnen
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Text einblenden
        cv2.putText(frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Bild anzeigen
        cv2.imshow("DGS Live Gesture Recognition (MLP)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()