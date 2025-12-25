import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import joblib
import os

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


# ------------------------------------------------------------
# Modell & LabelEncoder laden
# ------------------------------------------------------------

MODEL_PATH = "training/models/gesture_model_phoenix2.h5"
LABEL_ENCODER_PATH = "training/models/label_encoder_phoenix2.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    CLASSES = list(label_encoder.classes_)
else:
    CLASSES = ["Zwei", "Vier"]


# ------------------------------------------------------------
# MediaPipe HandLandmarker (Tasks API)
# ------------------------------------------------------------

HAND_MODEL_PATH = "training/models/hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=base_options.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ------------------------------------------------------------
# Feature-Extraktion
# ------------------------------------------------------------

def extract_keypoints_from_result(result):
    keypoints = []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])

    keypoints = np.array(keypoints).flatten()

    if keypoints.shape[0] < 127:
        keypoints = np.pad(
            keypoints,
            (0, 127 - keypoints.shape[0]),
            mode="constant"
        )

    return keypoints


# ------------------------------------------------------------
# Kamera & FPS
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)
prev_time = 0


# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    gesture_text = "Keine Hand erkannt"

    # --------------------------------------------------------
    # Landmark-Zeichnung + Inferenz
    # --------------------------------------------------------

    if result.hand_landmarks:
        h, w, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        features = extract_keypoints_from_result(result).reshape(1, -1)

        probs = model.predict(features, verbose=0)[0]
        pred_class = np.argmax(probs)

        top_3_indices = np.argsort(probs)[-3:][::-1]

        confidence_threshold = 0.5
        if probs[pred_class] > confidence_threshold:
            gesture_text = f"Zeichen: {CLASSES[pred_class]} ({probs[pred_class]*100:.1f}%)"
        else:
            gesture_text = "Unsicher (zu geringe Konfidenz)"

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # --------------------------------------------------------
    # Overlay Text
    # --------------------------------------------------------

    cv2.putText(frame, gesture_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if result.hand_landmarks:
        y_pos = 90
        for idx in top_3_indices:
            text = f"{CLASSES[idx]}: {probs[idx]*100:.1f}%"
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_pos += 30

    cv2.imshow("DGS Live Gesture Recognition (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
