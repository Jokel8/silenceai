"""
keypoints_normalize_processor.py
Handles normalization of hand keypoints
Pure processing logic, independent of I/O
"""

import numpy as np


class KeypointsNormalizeProcessor:
    """Processor for normalizing hand keypoints"""

    def relative_to_wrist_normalize(self, points_dict):
        """
        Normalize keypoints relative to wrist position for both hands
        
        Args:
            points_dict: Dictionary with 'left_hand' and 'right_hand' arrays
                        Each array contains 63 values (21 keypoints × 3 coordinates)
        
        Returns:
            Dictionary with normalized arrays
        """
        normalized = {}
        for hand in ['left_hand', 'right_hand']:
            points_array = points_dict[hand]
            if np.any(points_array != 0):
                # Reshape to (21, 3) for easier processing
                points = points_array.reshape(21, 3)
                # First point is wrist (index 0)
                wrist = points[0]
                # Normalize relative to wrist position
                normalized[hand] = (points - wrist).flatten()
            else:
                normalized[hand] = points_array
        return normalized

    def global_minmax_normalize(self, points_dict):
        """
        Perform global min-max normalization on both hands
        
        Args:
            points_dict: Dictionary with 'left_hand' and 'right_hand' arrays
                        Each array contains 63 values (21 keypoints × 3 coordinates)
        
        Returns:
            Dictionary with normalized arrays (scaled to [-1, 1] range approximately)
        """
        normalized = {}
        for hand in ['left_hand', 'right_hand']:
            points_array = points_dict[hand]
            max_abs_value = np.abs(points_array).max()
            if max_abs_value > 0:
                normalized[hand] = points_array / max_abs_value
            else:
                normalized[hand] = points_array
        return normalized

    def process(self, keypoints_dict):
        """
        Apply all normalization steps in sequence
        
        Args:
            keypoints_dict: Dictionary with 'left_hand' and 'right_hand' arrays
        
        Returns:
            Fully normalized keypoints dictionary
        """
        # Step 1: Normalize relative to wrist
        normalized = self.relative_to_wrist_normalize(keypoints_dict)
        # Step 2: Apply global min-max normalization
        normalized = self.global_minmax_normalize(normalized)
        return normalized
