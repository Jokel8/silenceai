import cv2
import mediapipe as mp
import argparse
import os

# Mediapipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# CLI: optional camera index
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--camera', '-c', type=int, default=int(os.environ.get('CAMERA_INDEX', 0)),
                    help='Camera index to use (default from CAMERA_INDEX env or 0)')
args, _ = parser.parse_known_args()
camera_index = args.camera

# Try opening the camera with DirectShow on Windows first (better compatibility)
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(camera_index)  # fallback

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    try:
        while True:
            if not cap.isOpened():
                print(f"Kamera mit Index {camera_index} konnte nicht geöffnet werden.")
                break

            ok, image = cap.read()
            if not ok or image is None:
                print("Kein Kamerabild empfangen oder Frame leer. Versuche erneut...")
                # short sleep to avoid tight loop (optional)
                cv2.waitKey(100)
                continue

            # Für Selfie-Ansicht optional spiegeln
            image = cv2.flip(image, 1)

            # In RGB konvertieren (für MediaPipe)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Hände erkennen
            results = hands.process(image_rgb)

            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Bild anzeigen
            cv2.imshow("Hand Tracking (MediaPipe)", image)

            # 'q' oder ESC beendet
            key = cv2.waitKey(5) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
