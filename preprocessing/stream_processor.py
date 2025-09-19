"""
stream_processor.py

Modular realtime pipeline for:
 - SelfieSegmentation + Hands + Pose -> safe foreground mask
 - CLAHE (contrast) + brightness on foreground
 - Preview windows for user
 - AI output: RGBA frames sized 210x260 at exactly 25 FPS
 - AI frames saved by worker (PNG) to disk
Usage:
  from stream_processor import StreamProcessor
  sp = StreamProcessor(out_dir="preprocessing/out")
  sp.start()
  ...
  sp.stop()
"""

import cv2
import numpy as np
import mediapipe as mp
import threading, queue, os, time
from typing import Optional

# ------------------- Defaults / Config -------------------
DEFAULT_AI_W = 210
DEFAULT_AI_H = 260   # requested size
DEFAULT_FPS = 25.0
DEFAULT_OUT_DIR = "preprocessing/out"

class StreamProcessor:
    def __init__(self,
                 camera_index:int = 0,
                 ai_out_dir: str = DEFAULT_OUT_DIR,
                 ai_w: int = DEFAULT_AI_W,
                 ai_h: int = DEFAULT_AI_H,
                 target_fps: float = DEFAULT_FPS,
                 ai_queue_max:int = 128,
                 seg_threshold:float = 0.4,
                 hand_conf:float = 0.5,
                 pose_conf:float = 0.5,
                 clahe_clip:float = 2.0,
                 clahe_tile=(8,8),
                 crop_padding:float = 1.08,
                 crop_min_frac:float = 0.42):
        self.camera_index = camera_index
        self.ai_out_dir = ai_out_dir
        os.makedirs(self.ai_out_dir, exist_ok=True)
        self.AI_W = ai_w
        self.AI_H = ai_h
        self.TARGET_FPS = target_fps
        self.FRAME_INTERVAL = 1.0 / target_fps
        self.ai_q = queue.Queue(maxsize=ai_queue_max)

        # MediaPipe params
        self.SEG_THRESHOLD = seg_threshold
        self.HAND_CONF = hand_conf
        self.POSE_CONF = pose_conf
        self.CLAHE_CLIP = clahe_clip
        self.CLAHE_TILE = clahe_tile
        self.CROP_PADDING = crop_padding
        self.CROP_MIN_FRAC = crop_min_frac

        # MediaPipe models (initialized on start)
        self.mp_selfie = None
        self.mp_hands = None
        self.mp_pose = None

        # runtime controls
        self._stop_event = threading.Event()
        self._thread = None
        self._ai_worker = None
        self._cap = None
        self._preview_frame = None
        self._brightness = 1.0

    # ------------------- Helper image functions (same logic as earlier) -------------------
    def _apply_clahe(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP, tileGridSize=self.CLAHE_TILE)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def _adjust_brightness(self, img_bgr, factor):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        l = l * factor
        l = np.clip(l, 0, 255)
        lab2 = cv2.merge((l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def _landmarks_to_mask(self, landmarks, image_shape):
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

    def _composite_on_checkerboard(self, bgr, alpha_255, tile=12):
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

    def _rgba_from_bgr_and_mask(self, bgr, mask_255):
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask_255
        return bgra

    def _crop_and_resize_by_mask(self, img, mask, target_w=None, target_h=None, padding=None, min_frac=None):
        # default to object's config
        if target_w is None: target_w = self.AI_W
        if target_h is None: target_h = self.AI_H
        if padding is None: padding = self.CROP_PADDING
        if min_frac is None: min_frac = self.CROP_MIN_FRAC
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
        crop_img = img[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()
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
        return rgba, {'x1':x1,'y1':y1,'x2':x2,'y2':y2}

    # ------------------- AI saver worker -------------------
    def _ai_saver_worker(self, stop_event):
        idx = 0
        while not stop_event.is_set() or not self.ai_q.empty():
            try:
                rgba = self.ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            path = os.path.join(self.ai_out_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(path, rgba)
            idx += 1

    # ------------------- Start / Stop -------------------
    def start(self, show_preview: bool = True):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Already running")
        # init mediapipe
        self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                                 min_detection_confidence=self.HAND_CONF,
                                                 min_tracking_confidence=self.HAND_CONF)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                              min_detection_confidence=self.POSE_CONF,
                                              min_tracking_confidence=self.POSE_CONF)
        self._cap = cv2.VideoCapture(self.camera_index)
        self._stop_event.clear()
        # ai saver
        self._ai_worker = threading.Thread(target=self._ai_saver_worker, args=(self._stop_event,), daemon=True)
        self._ai_worker.start()
        # main loop thread
        self._thread = threading.Thread(target=self._main_loop, args=(show_preview,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ai_worker is not None:
            self._ai_worker.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    # ------------------- Public getter for preview -------------------
    def get_preview(self):
        # returns last preview frame (BGR) or None
        return None if self._preview_frame is None else self._preview_frame.copy()

    # ------------------- Main processing loop -------------------
    def _main_loop(self, show_preview: bool):
        frame_idx = 0
        next_push_time = time.time()
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # small pause and retry
                time.sleep(0.01)
                continue
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Selfie Seg
            seg_res = self.mp_selfie.process(frame_rgb)
            seg_mask = seg_res.segmentation_mask if seg_res and seg_res.segmentation_mask is not None else None
            if seg_mask is None:
                base_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                base_mask = (seg_mask > self.SEG_THRESHOLD).astype(np.uint8)

            # Hands
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            hands_res = self.mp_hands.process(frame_rgb)
            if hands_res.multi_hand_landmarks:
                for hand_landmarks in hands_res.multi_hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    if len(pts) >= 3:
                        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                        cv2.fillConvexPoly(hand_mask, hull, 1)

            # Torso / pose
            torso_mask = np.zeros((h, w), dtype=np.uint8)
            pose_res = self.mp_pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                indices = []
                for idx in [0,11,12,23,24]:
                    if idx < len(lm):
                        indices.append(lm[idx])
                if len(indices) >= 3:
                    torso_mask = self._landmarks_to_mask(indices, frame.shape)

            # combine + smooth
            combined = np.clip(base_mask + hand_mask + torso_mask, 0, 1).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            combined_255 = (combined * 255).astype(np.uint8)

            # contrast + brightness on person only
            clahe_img = self._apply_clahe(frame)
            bright_img = self._adjust_brightness(clahe_img, self._brightness)

            # produce AI RGBA (cropped & resized)
            crop_rgba, _coords = self._crop_and_resize_by_mask(bright_img, combined_255,
                                                               target_w=self.AI_W, target_h=self.AI_H)

            # Wait until next scheduled push time to ensure exact frame rate
            now = time.time()
            if now < next_push_time:
                time.sleep(next_push_time - now)
                now = next_push_time
            # push into AI queue (all frames at target fps)
            try:
                self.ai_q.put_nowait(crop_rgba)
            except queue.Full:
                # if queue is full, we drop (or you can block until space)
                pass
            next_push_time += self.FRAME_INTERVAL
            frame_idx += 1

            # Preview (show raw | mask | final(composite) )
            composite_preview = self._composite_on_checkerboard(bright_img, combined_255, tile=12)
            mask_vis = cv2.cvtColor(combined_255, cv2.COLOR_GRAY2BGR)
            mask_vis = cv2.bitwise_and(mask_vis, np.array([0,255,0], dtype=np.uint8))
            left = cv2.resize(frame, (320,240))
            mid = cv2.resize(mask_vis, (320,240))
            right = cv2.resize(composite_preview, (320,240))
            preview = np.hstack([left, mid, right])
            self._preview_frame = preview
            if show_preview:
                cv2.imshow("User preview: raw | mask | final (checkerboard)", preview)
                # quick view of AI frame rendered over white
                view_final = np.where((crop_rgba[:,:,3]==255)[:,:,None], crop_rgba[:,:,:3], np.full_like(crop_rgba[:,:,:3], 255))
                cv2.imshow("Final (AI crop over white)", view_final)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop_event.set()
                    break

    # small convenience: adjust brightness/clipping at runtime
    def set_brightness(self, v: float):
        self._brightness = float(max(0.0, v))

    def set_clahe_clip(self, v: float):
        self.CLAHE_CLIP = float(max(0.01, v))

# -- end of StreamProcessor
