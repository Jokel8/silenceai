"""
camera_manager.py
Handles camera capture and frame acquisition
"""

import cv2
import time


class CameraManager:
    """Manages camera capture operations"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self._cap = None
    
    def start(self):
        """Start camera capture"""
        if self._cap is not None:
            raise RuntimeError("Camera already started")
        
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
    
    def stop(self):
        """Stop camera capture"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def read_frame(self):
        """
        Read a frame from the camera
        
        Returns:
            tuple: (success, frame) or (False, None) if failed
        """
        if self._cap is None:
            return False, None
        
        ok, frame = self._cap.read()
        if not ok:
            time.sleep(0.01)
            return False, None
        
        return True, frame
    
    def is_opened(self):
        """Check if camera is opened"""
        return self._cap is not None and self._cap.isOpened()
