"""
mask_processor.py
Handles mask creation and manipulation operations
"""

import cv2
import numpy as np


class MaskProcessor:
    """Handles mask creation from landmarks and mask operations"""
    
    @staticmethod
    def landmarks_to_mask(landmarks, image_shape):
        """Convert MediaPipe landmarks to binary mask"""
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
    
    @staticmethod
    def combine_masks(*masks):
        """Combine multiple masks with clipping"""
        combined = np.zeros_like(masks[0])
        for mask in masks:
            combined = np.clip(combined + mask, 0, 1)
        return combined.astype(np.uint8)
    
    @staticmethod
    def smooth_mask(mask, kernel_size=(7, 7)):
        """Smooth mask using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
