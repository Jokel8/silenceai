import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "rawData", "basicZweiVier")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Parameter
IMG_SIZE = (128, 128)  # wird nur für MediaPipe verarbeitet
BATCH_SIZE = 32
EPOCHS = 30

mp_hands = mp.solutions.hands
# create a reusable Hands instance for preprocessing
_hands_processor = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_keypoints(image, hands_processor=_hands_processor):
    """Keypoints aus einem Bild extrahieren"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_processor.process(rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None  # keine Hand erkannt

# Daten vorbereiten
X = []
y = []

classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
for label in classes:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, IMG_SIZE)
        keypoints = extract_keypoints(image)
        if keypoints is not None:
            X.append(keypoints)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Labels in Zahlen umwandeln
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# MLP-Modell definieren
model = models.Sequential([
    layers.Input(shape=(63,)),  # 21 Handpunkte × 3
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tensorboard_cb]
)

# Modell speichern
model_path = os.path.join(MODEL_DIR, "gesture_model.h5")
model.save(model_path)
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"Training abgeschlossen. Modell gespeichert als {model_path}")
print("Klassen:", le.classes_)
