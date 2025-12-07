"""
brightness_processor.py
Handles brightness adjustment operations
"""

import cv2
import numpy as np


class BrightnessProcessor:
    """Handles brightness adjustment for images"""
    
    def __init__(self, initial_factor=1.0):
        self._brightness_factor = max(0.0, float(initial_factor))
    
    def set_brightness(self, factor):
        """Set brightness factor (0.0 = black, 1.0 = normal, >1.0 = brighter)"""
        self._brightness_factor = max(0.0, float(factor))
    
    def get_brightness(self):
        """Get current brightness factor"""
        return self._brightness_factor
    
    def process(self, img_bgr):
        """Apply brightness adjustment to BGR image"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        l = l * self._brightness_factor
        l = np.clip(l, 0, 255)
        lab2 = cv2.merge((l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
