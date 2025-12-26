"""
mediapipe_processor.py
Handles all MediaPipe-related processing (segmentation, hands, pose)
"""

import cv2
import numpy as np
import mediapipe as mp
from mask_processor import MaskProcessor


class MediaPipeProcessor:
    """Handles MediaPipe model initialization and processing"""
    
    def __init__(self, seg_threshold=0.4, hand_conf=0.5, pose_conf=0.5):
        self.seg_threshold = seg_threshold
        self.hand_conf = hand_conf
        self.pose_conf = pose_conf
        
        # Initialize models
        self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=hand_conf,
            min_tracking_confidence=hand_conf
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=pose_conf,
            min_tracking_confidence=pose_conf
        )
        
        self.mask_processor = MaskProcessor()
    
    def process_segmentation(self, frame_rgb):
        """Process selfie segmentation and return base mask"""
        h, w = frame_rgb.shape[:2]
        seg_res = self.mp_selfie.process(frame_rgb)
        seg_mask = seg_res.segmentation_mask if seg_res and seg_res.segmentation_mask is not None else None
        
        if seg_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        
        return (seg_mask > self.seg_threshold).astype(np.uint8)
    
    def process_hands(self, frame_rgb):
        """Process hand detection and return hand mask"""
        h, w = frame_rgb.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        hands_res = self.mp_hands.process(frame_rgb)
        
        if hands_res.multi_hand_landmarks:
            for hand_landmarks in hands_res.multi_hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                if len(pts) >= 3:
                    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                    cv2.fillConvexPoly(hand_mask, hull, 1)
        
        return hand_mask
    
    def process_pose(self, frame_rgb, frame_shape):
        """Process pose detection and return torso mask"""
        h, w = frame_shape[:2]
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        pose_res = self.mp_pose.process(frame_rgb)
        
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            indices = []
            for idx in [0, 11, 12, 23, 24]:  # Key torso points
                if idx < len(lm):
                    indices.append(lm[idx])
            if len(indices) >= 3:
                torso_mask = self.mask_processor.landmarks_to_mask(indices, frame_shape)
        
        return torso_mask
    
    def process_all(self, frame_rgb, frame_shape):
        """
        Process all MediaPipe models an """