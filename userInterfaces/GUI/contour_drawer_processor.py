"""
contour_drawer_processor.py
Handles drawing contours and outlines on images
"""

import cv2


class ContourDrawerProcessor:
    """Handles drawing contours around masked regions"""
    
    def __init__(self, color=(0, 255, 0), thickness=3, overlay_thickness=8, overlay_alpha=0.22):
        self._color = color
        self._thickness = thickness
        self._overlay_thickness = overlay_thickness
        self._overlay_alpha = overlay_alpha
    
    def set_color(self, color):
        """Set contour color (BGR tuple)"""
        self._color = color
    
    def get_color(self):
        """Get current contour color"""
        return self._color
    
    def set_thickness(self, thickness):
        """Set main contour line thickness"""
        self._thickness = max(1, int(thickness))
    
    def get_thickness(self):
        """Get current thickness"""
        return self._thickness
    
    def set_overlay_thickness(self, thickness):
        """Set overlay contour thickness for glow effect"""
        self._overlay_thickness = max(1, int(thickness))
    
    def get_overlay_thickness(self):
        """Get current overlay thickness"""
        return self._overlay_thickness
    
    def set_overlay_alpha(self, alpha):
        """Set overlay transparency (0.0-1.0)"""
        self._overlay_alpha = max(0.0, min(1.0, float(alpha)))
    
    def get_overlay_alpha(self):
        """Get current overlay alpha"""
        return self._overlay_alpha
    
    def process(self, img, mask_255):
        """
        Draw contours on image based on mask
        Returns: image with contours drawn
        """
        result = img.copy()
        contours, _ = cv2.findContours(
            mask_255.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return result
        
        # Draw main contour
        cv2.drawContours(result, contours, -1, self._color, thickness=self._thickness)
        
        # Draw subtle translucent thicker outline for animation/glow effect
        overlay = result.copy()
        cv2.drawContours(overlay, contours, -1, self._color, thickness=self._overlay_thickness)
        result = cv2.addWeighted(
            overlay,
            self._overlay_alpha,
            result,
            1.0 - self._overlay_alpha,
            0
        )
        
        return result
