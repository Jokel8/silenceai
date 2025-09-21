import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import joblib
import os

# Modell und LabelEncoder laden
MODEL_PATH = "silenceai/Training/models/gesture_model_phoenix1.h5"
LABEL_ENCODER_PATH = "silenceai/Training/models/label_encoder_phoenix.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

# LabelEncoder laden (falls vorhanden), sonst Klassen manuell setzen
if os.path.exists(LABEL_ENCODER_PATH):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    CLASSES = list(label_encoder.classes_)
else:
    CLASSES = ["Zwei", "Vier"]  # Fallback

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# FPS Berechnung
prev_time = 0

def extract_keypoints(hand_landmarks):
    """Keypoints in 1D-Vektor umwandeln"""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

def extract_keypoints_from_results(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    # Auf 127 Werte auffüllen
    keypoints = np.array(keypoints).flatten()
    if keypoints.shape[0] < 127:
        keypoints = np.pad(keypoints, (0, 127 - keypoints.shape[0]), mode='constant')
    return keypoints

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
            features = extract_keypoints_from_results(results).reshape(1, -1)
            print(f"Feature values min/max: {features.min():.3f}/{features.max():.3f}")
            
            probs = model.predict(features, verbose=0)[0]
            pred_class = np.argmax(probs)
            
            # Debugging hinzufügen
            print(f"Top 3 Vorhersagen:")
            top_3_indices = np.argsort(probs)[-3:][::-1]
            for idx in top_3_indices:
                print(f"Klasse {CLASSES[idx]}: {probs[idx]*100:.2f}%")
            
            # Nur Vorhersagen über einem Schwellenwert anzeigen
            confidence_threshold = 0.5  # 50% Schwellenwert
            if probs[pred_class] > confidence_threshold:
                gesture_text = f"Zeichen: {CLASSES[pred_class]} ({probs[pred_class]*100:.1f}%)"
            else:
                gesture_text = "Unsicher (zu geringe Konfidenz)"

        # FPS berechnen
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Text einblenden
        cv2.putText(frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Top 3 Vorhersagen auf dem Bildschirm anzeigen
        if results.multi_hand_landmarks:
            y_pos = 90  # Startposition für die Top 3 Liste
            for idx in top_3_indices:
                text = f"{CLASSES[idx]}: {probs[idx]*100:.1f}%"
                cv2.putText(frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                y_pos += 30  # Abstand zwischen den Zeilen

        # Bild anzeigen
        cv2.imshow("DGS Live Gesture Recognition (MLP)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()