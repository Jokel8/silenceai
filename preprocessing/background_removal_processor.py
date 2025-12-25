"""
background_removal_processor.py
Handles background removal and alpha compositing
"""

import cv2
import numpy as np


class BackgroundRemovalProcessor:
    """Handles background removal and alpha channel operations"""
    
    def __init__(self):
        pass
    
    def create_rgba(self, bgr, mask_255):
        """Create RGBA image from BGR image and alpha mask"""
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask_255
        return bgra
    
    def composite_on_background(self, foreground_bgr, mask_255, background_bgr):
        """
        Composite foreground on background using mask
        Returns: composited BGR image
        """
        alpha = (mask_255.astype(np.float32) / 255.0)[:, :, None]
        result = (
            foreground_bgr.astype(np.float32) * alpha +
            background_bgr.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        return result
    
    def composite_on_checkerboard(self, bgr, alpha_255, tile_size=12):
        """Composite image with alpha channel on checkerboard pattern"""
        h, w = alpha_255.shape[:2]
        cb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                    cb[y:y+tile_size, x:x+tile_size] = (200, 200, 200)
                else:
                    cb[y:y+tile_size, x:x+tile_size] = (120, 120, 120)
        
        alpha = (alpha_255.astype(np.float32) / 255.0)[:, :, None]
        composed = (
            bgr.astype(np.float32) * alpha +
            cb.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        return composed
