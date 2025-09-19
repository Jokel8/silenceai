"""
Realtime webcam pipeline:
 - SelfieSegmentation + Hands + Pose -> sichere Vordergrund-Maske (wie vorher)
 - CLAHE (Kontrast) + Helligkeit auf Vordergrund
 - Preview-Fenster: raw | mask | final (transparent über Checkerboard)
 - AI-Stream: RGBA frames (210x300) put into a queue (example worker saves PNGs)
 Controls:
  - q: quit
  - + / - : Helligkeit erhöhen / verringern
  - ] / [: CLAHE clipLimit erhöhen / verringern
  - p: pause/unpause stream
"""
import cv2
import numpy as np
import mediapipe as mp
import threading, queue, os, time

# ------------------- Konfiguration -------------------
SEG_THRESHOLD = 0.4
HAND_CONF = 0.5
POSE_CONF = 0.5
MORPH_KERNEL = (7,7)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8,8)

BRIGHTNESS = 1.0  # multiplicative factor (>=0)
BRIGHTNESS_STEP = 0.05
CLAHE_STEP = 0.2

AI_QUEUE_MAX = 8
AI_OUTPUT_DIR = "preprocessing/out"  # Beispiel: worker schreibt PNGs dorthin
os.makedirs(AI_OUTPUT_DIR, exist_ok=True)

# Target AI resolution
AI_W = 210
AI_H = 300
CROP_PADDING = 1.08
CROP_MIN_FRAC = 0.42
frame_idx = 0

# ------------------- MediaPipe init -------------------
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                   min_detection_confidence=HAND_CONF, min_tracking_confidence=HAND_CONF)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                 min_detection_confidence=POSE_CONF, min_tracking_confidence=POSE_CONF)

# ------------------- Helferfunktionen -------------------
def apply_clahe_on_image(img_bgr, clip=2.0, tile=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def adjust_brightness_lab(img_bgr, factor=1.0):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l = l * factor
    l = np.clip(l, 0, 255)
    lab2 = cv2.merge((l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def landmarks_to_mask(landmarks, image_shape):
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
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask

def composite_on_checkerboard(bgr, alpha_255, tile=12):
    h, w = alpha_255.shape[:2]
    cb = np.zeros((h, w, 3), dtype=np.uint8)
    s = tile
    for y in range(0, h, s):
        for x in range(0, w, s):
            if ((x//s) + (y//s)) % 2 == 0:
                cb[y:y+s, x:x+s] = (200,200,200)
            else:
                cb[y:y+s, x:x+s] = (120,120,120)
    alpha = (alpha_255.astype(np.float32) / 255.0)[:, :, None]
    composed = (bgr.astype(np.float32) * alpha + cb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return composed

def rgba_from_bgr_and_mask(bgr, mask_255):
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_255
    return bgra

def crop_and_resize_by_mask(img, mask, target_w=AI_W, target_h=AI_H, padding=CROP_PADDING, min_frac=CROP_MIN_FRAC):
    ih, iw = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        tar_aspect = target_w / target_h
        if iw / ih >= tar_aspect:
            crop_h = ih
            crop_w = int(round(crop_h * tar_aspect))
        else:
            crop_w = iw
            crop_h = int(round(crop_w / tar_aspect))
        x1 = (iw - crop_w) // 2
        y1 = (ih - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
    else:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        new_w = bbox_w * padding
        new_h = bbox_h * padding
        min_side = int(round(min(iw, ih) * min_frac))
        if new_w < min_side:
            new_w = min_side
        if new_h < min_side:
            new_h = min_side
        tar_aspect = target_w / target_h
        if (new_w / new_h) < tar_aspect:
            new_w = new_h * tar_aspect
        else:
            new_h = new_w / tar_aspect
        new_w = min(new_w, iw)
        new_h = min(new_h, ih)
        x1 = int(round(cx - new_w / 2.0))
        y1 = int(round(cy - new_h / 2.0))
        x1 = max(0, min(x1, iw - int(round(new_w))))
        y1 = max(0, min(y1, ih - int(round(new_h))))
        x2 = x1 + int(round(new_w))
        y2 = y1 + int(round(new_h))
    crop_coords = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
    crop_img = img[crop_coords['y1']:crop_coords['y2'], crop_coords['x1']:crop_coords['x2']].copy()
    crop_mask = mask[crop_coords['y1']:crop_coords['y2'], crop_coords['x1']:crop_coords['x2']].copy()
    ch, cw = crop_img.shape[:2]
    if cw == 0 or ch == 0:
        crop_img = img.copy()
        crop_mask = mask.copy()
        cw, ch = crop_img.shape[1], crop_img.shape[0]
    scale = target_w / float(cw)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_img = cv2.resize(crop_img, (target_w, target_h), interpolation=interp)
    resized_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    rgba = cv2.cvtColor(resized_img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = resized_mask
    return rgba, crop_coords

# ------------------- AI worker (Beispiel: PNG speichern) -------------------
def ai_worker_func(q: queue.Queue, stop_event: threading.Event):
    idx = 0
    while not stop_event.is_set():
        try:
            rgba = q.get(timeout=0.2)
        except queue.Empty:
            continue
        path = os.path.join(AI_OUTPUT_DIR, f"frame_{int(time.time()*1000)}_{idx:04d}.png")
        cv2.imwrite(path, rgba)
        idx += 1

# ------------------- Main Loop -------------------
def main():
    global CLAHE_CLIP, BRIGHTNESS
    cap = cv2.VideoCapture(0)
    ai_q = queue.Queue(maxsize=AI_QUEUE_MAX)
    stop_event = threading.Event()
    ai_thread = threading.Thread(target=ai_worker_func, args=(ai_q, stop_event), daemon=True)
    ai_thread.start()

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kein Frame, Ende.")
                break
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Selfie-Segmentation
            seg_res = mp_selfie.process(frame_rgb)
            seg_mask = seg_res.segmentation_mask if seg_res and seg_res.segmentation_mask is not None else None
            if seg_mask is None:
                base_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                base_mask = (seg_mask > SEG_THRESHOLD).astype(np.uint8)

            # Hands
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            hands_res = mp_hands.process(frame_rgb)
            if hands_res.multi_hand_landmarks:
                for hand_landmarks in hands_res.multi_hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    if len(pts) >= 3:
                        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                        cv2.fillConvexPoly(hand_mask, hull, 1)

            # Torso/Pose
            torso_mask = np.zeros((h, w), dtype=np.uint8)
            pose_res = mp_pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                indices = []
                for idx in [0, 11, 12, 23, 24]:
                    if idx < len(lm):
                        indices.append(lm[idx])
                if len(indices) >= 3:
                    torso_mask = landmarks_to_mask(indices, frame.shape)

            # combine + smooth
            combined = np.clip(base_mask + hand_mask + torso_mask, 0, 1).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            combined_255 = (combined * 255).astype(np.uint8)

            # contrast + brightness on person only
            clahe_img = apply_clahe_on_image(frame, clip=CLAHE_CLIP, tile=CLAHE_TILE)
            bright_img = adjust_brightness_lab(clahe_img, BRIGHTNESS)

            # --- Crop & resize for AI consumer (210x300) ---
            crop_rgba, crop_coords = crop_and_resize_by_mask(bright_img, combined_255,
                                                             target_w=AI_W, target_h=AI_H,
                                                             padding=CROP_PADDING, min_frac=CROP_MIN_FRAC)
            # Push only every 10th frame
            if (frame_idx % 10) == 0:
                try:
                    ai_q.put_nowait(crop_rgba.copy())
                except queue.Full:
                    pass

            frame_idx += 1

            # user preview
            composite_preview = composite_on_checkerboard(bright_img, combined_255, tile=12)
            mask_vis = cv2.cvtColor(combined_255, cv2.COLOR_GRAY2BGR)
            mask_vis = cv2.bitwise_and(mask_vis, np.array([0,255,0], dtype=np.uint8))
            left = cv2.resize(frame, (320,240))
            mid = cv2.resize(mask_vis, (320,240))
            right = cv2.resize(composite_preview, (320,240))
            preview = np.hstack([left, mid, right])
            cv2.imshow("User preview: raw | mask | final (checkerboard)", preview)

            # view final (RGBA over white for quick view)
            view_final = np.where((crop_rgba[:,:,3]==255)[:,:,None], crop_rgba[:,:,:3], np.full_like(crop_rgba[:,:,:3], 255))
            cv2.imshow("Final (AI crop 210x300 over white)", view_final)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                BRIGHTNESS += BRIGHTNESS_STEP
                print(f"Helligkeit -> {BRIGHTNESS:.2f}")
            elif key == ord('-') or key == ord('_'):
                BRIGHTNESS = max(0.1, BRIGHTNESS - BRIGHTNESS_STEP)
                print(f"Helligkeit -> {BRIGHTNESS:.2f}")
            elif key == ord(']'):
                CLAHE_CLIP += CLAHE_STEP
                print(f"CLAHE clip -> {CLAHE_CLIP:.2f}")
            elif key == ord('['):
                CLAHE_CLIP = max(0.1, CLAHE_CLIP - CLAHE_STEP)
                print(f"CLAHE clip -> {CLAHE_CLIP:.2f}")
            elif key == ord('p'):
                print("Pause. Drücke eine Taste um fortzufahren...")
                cv2.waitKey(0)

    finally:
        stop_event.set()
        ai_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
