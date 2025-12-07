"""
image_processor.py
Handles all image processing operations (CLAHE, brightness, compositing)
"""

import cv2
import numpy as np


class ImageProcessor:
    """Handles image enhancement and manipulation operations"""
    
    def __init__(self, clahe_clip=2.0, clahe_tile=(8, 8)):
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
    
    def apply_clahe(self, img_bgr):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    
    def adjust_brightness(self, img_bgr, factor):
        """Adjust image brightness by factor"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        l = l * factor
        l = np.clip(l, 0, 255)
        lab2 = cv2.merge((l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    
    def composite_on_checkerboard(self, bgr, alpha_255, tile=12):
        """Composite image with alpha channel on checkerboard pattern"""
        h, w = alpha_255.shape[:2]
        cb = np.zeros((h, w, 3), dtype=np.uint8)
        s = tile
        for y in range(0, h, s):
            for x in range(0, w, s):
                if ((x//s) + (y//s)) % 2 == 0:
                    cb[y:y+s, x:x+s] = (200, 200, 200)
                else:
                    cb[y:y+s, x:x+s] = (120, 120, 120)
        alpha = (alpha_255.astype(np.float32) / 255.0)[:, :, None]
        composed = (bgr.astype(np.float32) * alpha + cb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return composed
    
    def rgba_from_bgr_and_mask(self, bgr, mask_255):
        """Create RGBA image from BGR image and alpha mask"""
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask_255
        return bgra
