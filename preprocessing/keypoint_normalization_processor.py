"""
keypoints_normalize_processor.py
Handles normalization of hand, lips, and face keypoints
Pure processing logic, independent of I/O
"""

import numpy as np


class KeypointNormalizationProcessor:
    """Processor for normalizing hand, lips, and face keypoints"""

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

    def _compute_eye_distance(self, left_eye, right_eye):
        """
        Compute normalized eye distance for feature normalization
        
        Args:
            left_eye: Array of shape (12,) representing left eye keypoints (4 landmarks × 3 coords)
            right_eye: Array of shape (12,) representing right eye keypoints (4 landmarks × 3 coords)
        
        Returns:
            float: Distance between eye centers, or 1.0 if eyes not detected
        """
        if np.all(left_eye == 0) or np.all(right_eye == 0):
            return 1.0

        # Reshape to get individual landmarks
        left_eye_pts = left_eye.reshape(4, 3)
        right_eye_pts = right_eye.reshape(4, 3)

        # Compute center of each eye
        left_center = left_eye_pts[:, :2].mean(axis=0)
        right_center = right_eye_pts[:, :2].mean(axis=0)

        # Compute Euclidean distance
        distance = np.linalg.norm(right_center - left_center)
        return max(distance, 1e-6)  # Avoid division by zero

    def normalize_face_features(self, lips_keypoints, left_eye, right_eye, head_pose):
        """
        Extract and normalize face features from face keypoints and head pose
        
        Args:
            lips_keypoints: Array of shape (54,) containing 18 lip landmarks × 3 coordinates
            left_eye: Array of shape (12,) for eye distance normalization
            right_eye: Array of shape (12,) for eye distance normalization
            head_pose: Dictionary with 'pitch', 'yaw', 'roll' angles in degrees
        
        Returns:
            Dictionary containing extracted and normalized face features:
                - 'lip_opening': Vertical distance between upper and lower lips
                - 'lip_width': Horizontal distance between mouth corners
                - 'aspect_ratio': Lip width / lip opening ratio
                - 'roundness': Indicator based on width-to-height ratio
                - 'raw_keypoints': Normalized lip keypoint coordinates
                - 'pitch': Head rotation up/down (degrees)
                - 'yaw': Head rotation left/right (degrees)
                - 'roll': Head tilt (degrees)
        """
        if np.all(lips_keypoints == 0):
            return {
                'lip_opening': 0.0,
                'lip_width': 0.0,
                'aspect_ratio': 0.0,
                'roundness': 0.0,
                'raw_keypoints': lips_keypoints,
                'pitch': head_pose.get('pitch', 0.0),
                'yaw': head_pose.get('yaw', 0.0),
                'roll': head_pose.get('roll', 0.0)
            }

        # Reshape to get individual landmarks (18 landmarks × 3 coords)
        lips_pts = lips_keypoints.reshape(18, 3)

        # Get eye distance for normalization
        eye_distance = self._compute_eye_distance(left_eye, right_eye)

        # Indices for key lip landmarks
        # Upper lip: indices 0-8 (9 points)
        # Lower lip: indices 9-17 (9 points)
        upper_lip_center_y = lips_pts[0:9, 1].mean()
        lower_lip_center_y = lips_pts[9:18, 1].mean()

        # Mouth corners: first and last points of upper lip (indices 0 and 8)
        mouth_left_x = lips_pts[0, 0]
        mouth_right_x = lips_pts[8, 0]

        # 1. Lip opening (vertical distance)
        lip_opening = abs(lower_lip_center_y - upper_lip_center_y)
        lip_opening_normalized = lip_opening / eye_distance

        # 2. Lip width (horizontal distance between corners)
        lip_width = abs(mouth_right_x - mouth_left_x)
        lip_width_normalized = lip_width / eye_distance

        # 3. Aspect ratio (width / opening)
        aspect_ratio = lip_width_normalized / max(lip_opening_normalized, 1e-6)

        # 4. Roundness indicator (based on curvature and width-to-height ratio)
        # Simple approximation: how much the lips deviate from a rectangle
        # Values closer to 1 indicate rounder lips
        roundness = min(1.0, lip_opening_normalized / max(lip_width_normalized, 1e-6))

        # Normalize relative coordinates
        # Normalize each point relative to mouth center and eye distance
        mouth_center_x = (mouth_left_x + mouth_right_x) / 2.0
        mouth_center_y = (upper_lip_center_y + lower_lip_center_y) / 2.0

        normalized_keypoints = []
        for pt in lips_pts:
            x_norm = (pt[0] - mouth_center_x) / eye_distance
            y_norm = (pt[1] - mouth_center_y) / eye_distance
            z_norm = pt[2] / eye_distance if eye_distance > 0 else pt[2]
            normalized_keypoints.extend([x_norm, y_norm, z_norm])

        return {
            'lip_opening': float(lip_opening_normalized),
            'lip_width': float(lip_width_normalized),
            'aspect_ratio': float(aspect_ratio),
            'roundness': float(roundness),
            'raw_keypoints': np.array(normalized_keypoints),
            'pitch': float(head_pose.get('pitch', 0.0)),
            'yaw': float(head_pose.get('yaw', 0.0)),
            'roll': float(head_pose.get('roll', 0.0))
        }

    def normalize_lips_features(self, lips_keypoints, left_eye, right_eye):
        """
        Legacy method - wrapper around normalize_face_features for backwards compatibility
        """
        return self.normalize_face_features(lips_keypoints, left_eye, right_eye, {})

    def process(self, keypoints_dict):
        """
        Apply all normalization steps in sequence
        
        Args:
            keypoints_dict: Dictionary with optional keys:
                - 'left_hand': Hand keypoints (63 values)
                - 'right_hand': Hand keypoints (63 values)
                - 'lips': Lips keypoints (54 values) [deprecated, use 'face_lips']
                - 'face_lips': Lips keypoints (54 values)
                - 'left_eye': Left eye keypoints (12 values)
                - 'right_eye': Right eye keypoints (12 values)
                - 'head_pose': Dict with pitch, yaw, roll angles
        
        Returns:
            Dictionary with normalized keypoints and features
        """
        normalized = {}

        # Normalize hand keypoints if present
        if 'left_hand' in keypoints_dict or 'right_hand' in keypoints_dict:
            hand_dict = {
                'left_hand': keypoints_dict.get('left_hand', np.zeros(63)),
                'right_hand': keypoints_dict.get('right_hand', np.zeros(63))
            }
            normalized_hands = self.relative_to_wrist_normalize(hand_dict)
            normalized_hands = self.global_minmax_normalize(normalized_hands)
            normalized.update(normalized_hands)

        # Normalize face features if present (includes lips and head pose)
        if 'face_lips' in keypoints_dict or 'lips' in keypoints_dict:
            lips_keypoints = keypoints_dict.get('face_lips', keypoints_dict.get('lips', np.zeros(54)))
            left_eye = keypoints_dict.get('left_eye', np.zeros(12))
            right_eye = keypoints_dict.get('right_eye', np.zeros(12))
            head_pose = keypoints_dict.get('head_pose', {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0})

            face_features = self.normalize_face_features(lips_keypoints, left_eye, right_eye, head_pose)
            normalized['face'] = face_features
            # Also keep backwards compatibility
            normalized['lips'] = face_features

        return normalized

