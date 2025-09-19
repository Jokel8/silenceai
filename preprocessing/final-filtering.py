import cv2
import mediapipe as mp
import numpy as np

# --- Konfiguration / Hyperparameter (anpassbar) ---
SEG_THRESHOLD = 0.4          # niedriger -> mehr Vordergrund aus Segmenter
HAND_CONF = 0.5              # Mindestvertrauen für Hand-Detektion
POSE_CONF = 0.5              # Mindestvertrauen für Pose-Detektion
MORPH_KERNEL = (7, 7)        # Kernel für Morphologie (glättet Maske)
CLAHE_CLIP = 2.0             # CLAHE Parameter
CLAHE_TILE = (8, 8)

# --- MediaPipe initialisieren ---
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=HAND_CONF,
    min_tracking_confidence=HAND_CONF
)
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=POSE_CONF,
    min_tracking_confidence=POSE_CONF
)

cap = cv2.VideoCapture(0)

def landmarks_to_mask(landmarks, image_shape):
    """Konvertiert Normalisierte Landmarks (x,y in 0..1) zu einer gefüllten Maske (uint8)."""
    h, w = image_shape[:2]
    pts = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
    if len(pts) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    pts = np.array(pts, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    return mask

def apply_clahe_on_image(img_bgr):
    """Wendet CLAHE auf das ganze Bild an und gibt ein BGR Bild zurück."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    res = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return res

while True:
    success, frame = cap.read()
    if not success:
        print("No more stream :(")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1) Selfie-Segmenter Maske (float 0..1) ---
    seg_res = mp_selfie.process(frame_rgb)
    seg_mask = seg_res.segmentation_mask if seg_res and seg_res.segmentation_mask is not None else None
    if seg_mask is None:
        base_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    else:
        base_mask = (seg_mask > SEG_THRESHOLD).astype(np.uint8)  # binärmaske

    h, w = frame.shape[:2]

    # --- 2) Hand-Masken hinzufügen (explode/force foreground) ---
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    hands_res = mp_hands.process(frame_rgb)
    if hands_res.multi_hand_landmarks:
        for hand_landmarks in hands_res.multi_hand_landmarks:
            # Nutze alle Landmarks der Hand als Polygon (Konvexe Hülle sinnvoll)
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            if len(pts) >= 3:
                hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                cv2.fillConvexPoly(hand_mask, hull, 1)

    # --- 3) Torso/Oberkörper-Maske mittels Pose-Landmarks ---
    torso_mask = np.zeros((h, w), dtype=np.uint8)
    pose_res = mp_pose.process(frame_rgb)
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        # Wähle relevante Punkte: Schultern, Brust/Brustkorb, Hüfte, Nacken (falls vorhanden)
        indices = []
        # Laut MediaPipe Pose indices (common): 11 = left_shoulder, 12 = right_shoulder,
        # 23 = left_hip, 24 = right_hip, 0 = nose (als grobe top), 1 = left_eye_inner - nicht immer nötig
        for idx in [0, 11, 12, 23, 24]:  # nose + shoulders + hips
            if idx < len(lm):
                indices.append(lm[idx])
        if len(indices) >= 3:
            torso_mask = landmarks_to_mask(indices, frame.shape)

    # --- 4) Vereinigung Masken + Morphologische Glättung ---
    combined = np.clip(base_mask + hand_mask + torso_mask, 0, 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)  # kleine Flecken entfernen
    combined = (combined * 255).astype(np.uint8)  # 0/255 für bitwise mask

    # --- 5) Kontrast (CLAHE) nur für Vordergrund anwenden ---
    clahe_img = apply_clahe_on_image(frame)
    # final: wenn Mask==255 -> klahe_img, sonst -> weißer Hintergrund (oder transparent)
    bg_color = (255, 255, 255)  # weißer Hintergrund
    bg = np.full_like(frame, bg_color, dtype=np.uint8)

    mask_3ch = cv2.merge([combined, combined, combined])
    final = np.where(mask_3ch == 255, clahe_img, bg)

    # Optional: zeige auch die Maske und debug Fenster
    mask_vis = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    debug = np.hstack([frame, mask_vis, final])  # nebeneinander: original | maske | result

    cv2.imshow("Original | Mask | Final (CLAHE on person, white bg)", debug)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
