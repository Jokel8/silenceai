"""
stream_processor.py

StreamProcessor with preprocessing toggle:
 - preprocessing_enabled: if True -> background removed + CLAHE+brightness applied.
   Preview: raw background + processed person foreground + green outline.
   AI: processed RGBA crops (transparent background).
 - if False:
   Preview: raw only.
   AI: raw resized -> BGRA (alpha=255).
"""

import cv2
import numpy as np
import mediapipe as mp
import threading, queue, os, time
from typing import Optional
import importlib.util


# ------------------- Defaults / Config -------------------
DEFAULT_AI_W = 210
DEFAULT_AI_H = 300   # size, because got streched in trainingsdata
DEFAULT_FPS = 25.0

class Preprocessing:
    # ------------------- Public controls -------------------
    def __init__(self, state,
                 camera_index:int = 0,
                 ai_w: int = DEFAULT_AI_W,
                 ai_h: int = DEFAULT_AI_H,
                 target_fps: float = DEFAULT_FPS,
                 ai_out_dir: str = "preprocessing/out",
                 ai_queue_max:int = 128,
                 seg_threshold:float = 0.4,
                 hand_conf:float = 0.5,
                 pose_conf:float = 0.5,
                 clahe_clip:float = 2.0,
                 clahe_tile=(8,8),
                 crop_padding:float = 1.08,
                 crop_min_frac:float = 0.42):
        self.camera_index = camera_index
        self.AI_W = ai_w
        self.AI_H = ai_h
        self.TARGET_FPS = target_fps
        self.FRAME_INTERVAL = 1.0 / target_fps
        self.ai_q = queue.Queue(maxsize=ai_queue_max)

    def set_brightness(self, v: float):
        self._brightness = float(max(0.0, v))

    def set_clahe_clip(self, v: float):
        self.CLAHE_CLIP = float(max(0.01, v))

    # ------------------- Helper image functions -------------------
    def _apply_clahe(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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
        if target_w is None: target_w = self.AI_W
        if target_h is None: target_h = self.AI_H
        if padding is None: padding = 1.08
        if min_frac is None: min_frac = 0.42
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
