"""
hand_detection_processor.py
Handles MediaPipe hand detection and mask creation
"""

import cv2
import numpy as np
import mediapipe as mp


class HandDetectionProcessor:
    """Handles hand detection using MediaPipe"""
    
    def __init__(self, confidence=0.5, max_hands=2):
        self._confidence = confidence
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
    
    def set_confidence(self, confidence):
        """Set detection confidence threshold (0.0-1.0)"""
        self._confidence = max(0.0, min(1.0, float(confidence)))
        # Note: MediaPipe Hands needs to be recreated to apply new confidence
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self._confidence,
            min_tracking_confidence=self._confidence
        )
    
    def get_confidence(self):
        """Get current confidence threshold"""
        return self._confidence
    
    def process(self, frame_rgb):
        """
        Process frame and return hand mask
        Returns: binary mask (0 or 1)
        """
        h, w = frame_rgb.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        hands_res = self._mp_hands.process(frame_rgb)
        
        if hands_res.multi_hand_landmarks:
            for hand_landmarks in hands_res.multi_hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                if len(pts) >= 3:
                    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                    cv2.fillConvexPoly(hand_mask, hull, 1)
        
        return hand_mask
