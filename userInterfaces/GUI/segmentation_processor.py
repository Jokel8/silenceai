"""
segmentation_processor.py
Handles MediaPipe selfie segmentation
"""

import numpy as np
import mediapipe as mp


class SegmentationProcessor:
    """Handles selfie segmentation using MediaPipe"""
    
    def __init__(self, threshold=0.4, model_selection=1):
        self._threshold = threshold
        self._mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
    
    def set_threshold(self, threshold):
        """Set segmentation threshold (0.0-1.0)"""
        self._threshold = max(0.0, min(1.0, float(threshold)))
    
    def get_threshold(self):
        """Get current threshold"""
        return self._threshold
    
    def process(self, frame_rgb):
        """
        Process frame and return binary segmentation mask
        Returns: binary mask (0 or 1)
        """
        h, w = frame_rgb.shape[:2]
        seg_res = self._mp_selfie.process(frame_rgb)
        seg_mask = seg_res.segmentation_mask if seg_res and seg_res.segmentation_mask is not None else None
        
        if seg_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        
        return (seg_mask > self._threshold).astype(np.uint8)
