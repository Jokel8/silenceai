"""
mask_combiner_processor.py
Handles mask combining and smoothing operations
"""

import cv2
import numpy as np


class MaskCombinerProcessor:
    """Handles combining and smoothing multiple masks"""
    
    def __init__(self, kernel_size=(7, 7)):
        self._kernel_size = kernel_size
    
    def set_kernel_size(self, size):
        """Set morphological kernel size (tuple: width, height)"""
        self._kernel_size = size
    
    def get_kernel_size(self):
        """Get current kernel size"""
        return self._kernel_size
    
    def process(self, *masks):
        """
        Combine multiple masks and smooth them
        Returns: combined smoothed mask (0-255)
        """
        # Combine masks
        combined = np.zeros_like(masks[0])
        for mask in masks:
            combined = np.clip(combined + mask, 0, 1)
        combined = combined.astype(np.uint8)
        
        # Smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self._kernel_size)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Convert to 0-255 range
        combined_255 = (combined * 255).astype(np.uint8)
        
        return combined_255
