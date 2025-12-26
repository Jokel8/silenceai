"""
hand_detection_processor.py
Handles hand detection and mask creation (MediaPipe Tasks API)
Extended to support keypoint extraction for training and GUI use
"""

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class HandDetectionProcessor:
    """Handles hand detection using MediaPipe Tasks API"""

    def __init__(
        self,
        confidence=0.5,
        max_hands=2,
        model_path= "preprocessing/models/hand_landmarker.task",
        draw_keypoints=False
    ):
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._max_hands = max_hands
        self._model_path = model_path
        self._draw_keypoints = draw_keypoints

        self._create_landmarker()

    def _create_landmarker(self):
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=self._model_path
            ),
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._confidence,
            min_hand_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence
        )

        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def set_confidence(self, confidence):
        """Set detection confidence threshold (0.0–1.0)"""
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._create_landmarker()

    def get_confidence(self):
        """Get current confidence threshold"""
        return self._confidence

    def set_draw_keypoints(self, draw_keypoints):
        """Enable/disable keypoint visualization"""
        self._draw_keypoints = draw_keypoints

    def process(self, frame_rgb):
        """
        Process frame and return hand mask
        Returns: binary mask (0 or 1)
        """
        h, w = frame_rgb.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self._landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                pts = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand_landmarks
                ]

                if len(pts) >= 3:
                    hull = cv2.convexHull(
                        np.array(pts, dtype=np.int32)
                    )
                    cv2.fillConvexPoly(hand_mask, hull, 1)

        return hand_mask

    def extractKeypoints(self, frame):
        """
        Extract hand keypoints from frame (training/pipeline compatible)
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with keys:
                - "frame": annotated frame
                - "left_hand": numpy array of 63 values (21 keypoints × 3 coordinates)
                - "right_hand": numpy array of 63 values (21 keypoints × 3 coordinates)
        """
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self._landmarker.detect(mp_image)

        # Initialize empty arrays for both hands (21 keypoints × 3 coordinates)
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        # Process detected hands
        if result.hand_landmarks and result.handedness:
            h, w = frame.shape[:2]

            for hand_landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness
            ):
                label = handedness[0].category_name
                keypoints = []

                # Extract x, y, z coordinates for all 21 landmarks
                for lm in hand_landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])

                    # Optionally draw keypoints on frame
                    if self._draw_keypoints:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Assign to correct hand based on label
                if label == "Left":
                    left_hand = np.array(keypoints)
                elif label == "Right":
                    right_hand = np.array(keypoints)

        return {
            "frame": frame,
            "left_hand": left_hand,
            "right_hand": right_hand
        }