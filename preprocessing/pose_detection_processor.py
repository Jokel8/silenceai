"""
pose_detection_processor.py
Handles pose detection and torso mask creation (MediaPipe Tasks API)
"""

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class PoseDetectionProcessor:
    """Handles pose detection using MediaPipe Tasks API"""

    def __init__(
        self,
        confidence=0.5,
        model_path="silenceai/preprocessing/models/pose_landmarker_lite.task"
    ):
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._model_path = model_path
        self._create_landmarker()

    def _create_landmarker(self):
        options = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=self._model_path
            ),
            min_pose_detection_confidence=self._confidence,
            min_pose_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def set_confidence(self, confidence):
        """Set detection confidence threshold (0.0–1.0)"""
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._create_landmarker()

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

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self._landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Torso-relevante Landmark-Indizes
            # Nose = 0, Left Shoulder = 11, Right Shoulder = 12,
            # Left Hip = 23, Right Hip = 24
            indices = [0, 11, 12, 23, 24]
            selected = [
                landmarks[i] for i in indices if i < len(landmarks)
            ]

            if len(selected) >= 3:
                torso_mask = self._landmarks_to_mask(selected, (h, w))

        return torso_mask

    @staticmethod
    def _landmarks_to_mask(landmarks, image_shape):
        """Convert pose landmarks to binary mask"""
        h, w = image_shape
        pts = []

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))

        if len(pts) < 3:
            return np.zeros((h, w), dtype=np.uint8)

        pts = np.array(pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)

        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)

        return mask
