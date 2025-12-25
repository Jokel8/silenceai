"""
crop_resize_processor.py
Handles intelligent cropping and resizing based on masks
"""

import cv2
import numpy as np


class CropResizeProcessor:
    """Handles intelligent cropping and resizing operations"""
    
    def __init__(self, padding=1.08, min_frac=0.42):
        self._padding = padding
        self._min_frac = min_frac
    
    def set_padding(self, padding):
        """Set padding factor around detected object (1.0 = tight, >1.0 = looser)"""
        self._padding = max(1.0, float(padding))
    
    def get_padding(self):
        """Get current padding factor"""
        return self._padding
    
    def set_min_fraction(self, min_frac):
        """Set minimum crop fraction of image size (0.0-1.0)"""
        self._min_frac = max(0.0, min(1.0, float(min_frac)))
    
    def get_min_fraction(self):
        """Get current minimum fraction"""
        return self._min_frac
    
    def process(self, img, mask, target_w, target_h):
        """
        Crop and resize image based on mask with intelligent padding
        Returns: (rgba_image, crop_coordinates_dict)
        """
        ih, iw = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        
        if len(xs) == 0 or len(ys) == 0:
            # No mask content - use center crop
            x1, y1, x2, y2 = self._center_crop_coords(iw, ih, target_w, target_h)
        else:
            # Mask content exists - intelligent crop
            x1, y1, x2, y2 = self._intelligent_crop_coords(
                xs, ys, iw, ih, target_w, target_h
            )
        
        # Perform crop
        crop_img = img[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()
        
        ch, cw = crop_img.shape[:2]
        if cw == 0 or ch == 0:
            crop_img = img.copy()
            crop_mask = mask.copy()
            cw, ch = crop_img.shape[1], crop_img.shape[0]
        
        # Resize
        scale = target_w / float(cw)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_img = cv2.resize(crop_img, (target_w, target_h), interpolation=interp)
        resized_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Create RGBA
        rgba = cv2.cvtColor(resized_img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = resized_mask
        
        return rgba, {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    
    def _center_crop_coords(self, iw, ih, target_w, target_h):
        """Calculate center crop coordinates"""
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
        return x1, y1, x2, y2
    
    def _intelligent_crop_coords(self, xs, ys, iw, ih, target_w, target_h):
        """Calculate intelligent crop coordinates based on mask content"""
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        # Apply padding
        new_w = bbox_w * self._padding
        new_h = bbox_h * self._padding
        
        # Enforce minimum size
        min_side = int(round(min(iw, ih) * self._min_frac))
        new_w = max(new_w, min_side)
        new_h = max(new_h, min_side)
        
        # Maintain target aspect ratio
        tar_aspect = target_w / target_h
        if (new_w / new_h) < tar_aspect:
            new_w = new_h * tar_aspect
        else:
            new_h = new_w / tar_aspect
        
        # Clip to image bounds
        new_w = min(new_w, iw)
        new_h = min(new_h, ih)
        
        # Calculate final coordinates
        x1 = int(round(cx - new_w / 2.0))
        y1 = int(round(cy - new_h / 2.0))
        x1 = max(0, min(x1, iw - int(round(new_w))))
        y1 = max(0, min(y1, ih - int(round(new_h))))
        x2 = x1 + int(round(new_w))
        y2 = y1 + int(round(new_h))
        
        return x1, y1, x2, y2
