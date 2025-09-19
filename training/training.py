import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Parameter
DATA_DIR = "silenceai\Training\data"
IMG_SIZE = (128, 128)  # wird nur für MediaPipe verarbeitet
BATCH_SIZE = 32
EPOCHS = 30

mp_hands = mp.solutions.hands

def extract_keypoints(image):
    """Keypoints aus einem Bild extrahieren"""
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None  # keine Hand erkannt

# Daten vorbereiten
X = []
y = []

classes = sorted(os.listdir(DATA_DIR))
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

# Training
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Modell speichern
model.save("silenceai\Training\models\gesture_model.h5")
print("Training abgeschlossen. Modell gespeichert als mlp_gesture_model.h5")
print("Klassen:", le.classes_)
