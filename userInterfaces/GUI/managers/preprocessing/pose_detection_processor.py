"""
pose_detection_processor.py
Handles MediaPipe pose detection and torso mask creation
"""

import cv2
import numpy as np
import mediapipe as mp


class PoseDetectionProcessor:
    """Handles pose detection using MediaPipe"""
    
    def __init__(self, confidence=0.5, model_complexity=1):
        self._confidence = confidence
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
    
    def set_confidence(self, confidence):
        """Set detection confidence threshold (0.0-1.0)"""
        self._confidence = max(0.0, min(1.0, float(confidence)))
        # Note: MediaPipe Pose needs to be recreated to apply new confidence
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=self._confidence,
            min_tracking_confidence=self._confidence
        )
    
    def get_confidence(self):
        """Get current confidence threshold"""
        return self._confidence
    
    def process(self, frame_rgb):
        """
        Process frame and return torso mask
        Returns: binary mask (0 or 1)
        """
        h, w = frame_rgb.shape[:2]
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        pose_res = self._mp_pose.process(frame_rgb)
        
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            indices = []
            # Key torso points: nose, shoulders, hips
            for idx in [0, 11, 12, 23, 24]:
                if idx < len(lm):
                    indices.append(lm[idx])
            
            if len(indices) >= 3:
                torso_mask = self._landmarks_to_mask(indices, (h, w))
        
        return torso_mask
    
    @staticmethod
    def _landmarks_to_mask(landmarks, image_shape):
        """Convert MediaPipe landmarks to binary mask"""
        h, w = image_shape
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
