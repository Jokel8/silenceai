import cv2
import mediapipe as mp

# Mediapipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hilfsfunktion: prüfen, ob Hand offen ist
def is_hand_open(hand_landmarks):
    # Wir vergleichen Y-Koordinaten von Fingerspitzen mit deren Grundgelenken
    # (Finger offen = Spitze weiter oben als Gelenk)
    tip_ids = [8, 12, 16, 20]  # Indizes für Zeige-, Mittel-, Ring-, kleiner Finger
    for tip_id in tip_ids:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            return False
    return True

# Kamera starten
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kein Kamerabild erkannt.")
            break

        # Bild spiegeln für Selfie-Ansicht
        image = cv2.flip(image, 1)

        # In RGB konvertieren (für MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hände erkennen
        results = hands.process(image_rgb)

        gesture_text = "Keine Geste"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_hand_open(hand_landmarks):
                    gesture_text = "Hand offen erkannt"

        # Text auf Bild schreiben
        cv2.putText(image, gesture_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Bild anzeigen
        cv2.imshow("Hand Gesture Prototype", image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC zum Beenden
            break

cap.release()
cv2.destroyAllWindows()
