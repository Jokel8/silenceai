"""
segmentation_processor.py
Handles MediaPipe selfie segmentation (Tasks API)
"""

import numpy as np
import mediapipe as mp
import os

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class SegmentationProcessor:
    """Handles selfie segmentation using MediaPipe Tasks API"""

    def __init__(
        self,
        threshold=0.4,
        model_path="silenceai/preprocessing/models/selfie_segmenter.tflite"
    ):
        print("Current working directory:", os.getcwd())
        self._threshold = max(0.0, min(1.0, float(threshold)))

        options = vision.ImageSegmenterOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=model_path
            ),
            output_category_mask=False,
            output_confidence_masks=True
        )

        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def set_threshold(self, threshold):
        """Set segmentation threshold (0.0–1.0)"""
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

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self._segmenter.segment(mp_image)

        if not result.confidence_masks:
            return np.zeros((h, w), dtype=np.uint8)

        # Für Selfie-Segmentation ist Maske[0] = Person
        confidence_mask = result.confidence_masks[0].numpy_view()

        return (confidence_mask > self._threshold).astype(np.uint8)
