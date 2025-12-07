"""
clahe_processor.py
Handles CLAHE (Contrast Limited Adaptive Histogram Equalization) operations
"""

import cv2


class CLAHEProcessor:
    """Handles CLAHE contrast enhancement for images"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self._clip_limit = max(0.01, float(clip_limit))
        self._tile_grid_size = tile_grid_size
    
    def set_clip_limit(self, clip_limit):
        """Set CLAHE clip limit (higher = more contrast enhancement)"""
        self._clip_limit = max(0.01, float(clip_limit))
    
    def get_clip_limit(self):
        """Get current clip limit"""
        return self._clip_limit
    
    def set_tile_grid_size(self, size):
        """Set tile grid size (tuple: width, height)"""
        self._tile_grid_size = size
    
    def get_tile_grid_size(self):
        """Get current tile grid size"""
        return self._tile_grid_size
    
    def process(self, img_bgr):
        """Apply CLAHE to BGR image"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=self._tile_grid_size
        )
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
