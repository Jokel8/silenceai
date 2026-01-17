"""
face_detection_processor.py
Handles face detection and keypoint extraction (MediaPipe Face Mesh)
Extracts face landmarks, head orientation (pose), and lips for DGS recognition
Extended to support head angles (Pitch, Yaw, Roll)
"""

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class FaceDetectionProcessor:
    """Handles face detection and keypoint extraction using MediaPipe Face Mesh"""

    # MediaPipe Face Mesh indices for lips
    # Upper lip: 61, 185, 40, 39, 37, 267, 269, 270, 409
    # Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375
    LIPS_LANDMARKS = {
        'upper_lip': [61, 185, 40, 39, 37, 267, 269, 270, 409],
        'lower_lip': [146, 91, 181, 84, 17, 314, 405, 321, 375],
        'mouth_left': 61,
        'mouth_right': 291,
        'upper_lip_center': 12,
        'lower_lip_center': 13
    }

    # Key facial landmarks for pose estimation
    POSE_LANDMARKS = {
        'nose': 1,           # Nasespitze
        'left_eye': 33,      # Linkes Auge
        'right_eye': 263,    # Rechtes Auge
        'left_ear': 234,     # Linkes Ohr
        'right_ear': 454,    # Rechtes Ohr
        'mouth_left': 61,    # Linker Mundwinkel
        'mouth_right': 291,  # Rechter Mundwinkel
        'chin': 199          # Kinn
    }

    def __init__(
        self,
        confidence=0.5,
        model_path="preprocessing/models/face_landmarker.task",
        draw_keypoints=False
    ):
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._model_path = model_path
        self._draw_keypoints = draw_keypoints

        self._create_face_landmarker()

    def _create_face_landmarker(self):
        """Initialize MediaPipe Face Landmarker"""
        options = vision.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=self._model_path
            ),
            min_face_detection_confidence=self._confidence,
            min_face_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        self._face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def set_confidence(self, confidence):
        """Set detection confidence threshold (0.0–1.0)"""
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._create_face_landmarker()

    def get_confidence(self):
        """Get current confidence threshold"""
        return self._confidence

    def set_draw_keypoints(self, draw_keypoints):
        """Enable/disable keypoint visualization"""
        self._draw_keypoints = draw_keypoints

    def _estimate_head_pose(self, landmarks):
        """
        Estimate head pose (Pitch, Yaw, Roll) from facial landmarks
        
        Uses the transformation matrix returned by MediaPipe Face Mesh
        to estimate 3D head orientation
        
        Args:
            landmarks: List of face landmarks
        
        Returns:
            dict with 'pitch', 'yaw', 'roll' in degrees
        """
        try:
            # Use key points to estimate pose
            # Get coordinates for estimation
            nose = landmarks[self.POSE_LANDMARKS['nose']]
            left_eye = landmarks[self.POSE_LANDMARKS['left_eye']]
            right_eye = landmarks[self.POSE_LANDMARKS['right_eye']]
            mouth_left = landmarks[self.POSE_LANDMARKS['mouth_left']]
            mouth_right = landmarks[self.POSE_LANDMARKS['mouth_right']]
            chin = landmarks[self.POSE_LANDMARKS['chin']]

            # Create 3D points
            nose_3d = np.array([nose.x, nose.y, nose.z])
            left_eye_3d = np.array([left_eye.x, left_eye.y, left_eye.z])
            right_eye_3d = np.array([right_eye.x, right_eye.y, right_eye.z])
            mouth_left_3d = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
            mouth_right_3d = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
            chin_3d = np.array([chin.x, chin.y, chin.z])

            # Calculate vectors for pose estimation
            # Eye baseline (horizontal)
            eye_baseline = right_eye_3d - left_eye_3d
            eye_baseline_norm = np.linalg.norm(eye_baseline)
            if eye_baseline_norm > 1e-6:
                eye_baseline = eye_baseline / eye_baseline_norm

            # Mouth baseline (horizontal)
            mouth_baseline = mouth_right_3d - mouth_left_3d
            mouth_baseline_norm = np.linalg.norm(mouth_baseline)
            if mouth_baseline_norm > 1e-6:
                mouth_baseline = mouth_baseline / mouth_baseline_norm

            # Vertical vector (eye to mouth)
            vertical = (mouth_left_3d + mouth_right_3d) / 2.0 - (left_eye_3d + right_eye_3d) / 2.0
            vertical_norm = np.linalg.norm(vertical)
            if vertical_norm > 1e-6:
                vertical = vertical / vertical_norm

            # Depth vector (chin to nose trend)
            depth = chin_3d - nose_3d
            depth_norm = np.linalg.norm(depth)
            if depth_norm > 1e-6:
                depth = depth / depth_norm

            # Calculate angles
            # Pitch (up/down): based on vertical component of nose relative to eyes
            eye_center = (left_eye_3d + right_eye_3d) / 2.0
            nose_to_eye = eye_center - nose_3d
            pitch_rad = np.arctan2(nose_to_eye[1], np.sqrt(nose_to_eye[0]**2 + nose_to_eye[2]**2))
            pitch = np.degrees(pitch_rad)

            # Yaw (left/right): based on horizontal nose position relative to eyes
            yaw_rad = np.arctan2(nose_to_eye[0], nose_to_eye[2])
            yaw = np.degrees(yaw_rad)

            # Roll (tilt): based on eye line angle
            eye_slope = (right_eye_3d[1] - left_eye_3d[1]) / max(eye_baseline_norm, 1e-6)
            roll_rad = np.arctan(eye_slope)
            roll = np.degrees(roll_rad)

            return {
                'pitch': float(pitch),
                'yaw': float(yaw),
                'roll': float(roll)
            }
        except Exception:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

    def extractKeypoints(self, frame):
        """
        Extract face keypoints from frame (training/pipeline compatible)
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with keys:
                - "frame": annotated frame
                - "lips": numpy array with lip keypoint coordinates
                - "left_eye": left eye keypoints (for distance normalization)
                - "right_eye": right eye keypoints (for distance normalization)
                - "head_pose": dict with pitch, yaw, roll angles (in degrees)
                - "face_landmarks": all 468 face landmarks as array
        """
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self._face_landmarker.detect(mp_image)

        h, w = frame.shape[:2]

        # Initialize empty arrays
        lips_keypoints = np.zeros(54)  # 18 landmarks × 3 coordinates
        left_eye = np.zeros(12)        # 4 eye corners × 3 coordinates
        right_eye = np.zeros(12)       # 4 eye corners × 3 coordinates
        face_landmarks_array = np.zeros(468 * 3)  # All 468 landmarks × 3 coords
        head_pose = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

        if result.face_landmarks:
            # Process first face detected
            landmarks = result.face_landmarks[0]

            # Extract all face landmarks
            all_landmarks_list = []
            for lm in landmarks:
                all_landmarks_list.extend([lm.x, lm.y, lm.z])
            if all_landmarks_list:
                face_landmarks_array = np.array(all_landmarks_list)

            # Estimate head pose
            head_pose = self._estimate_head_pose(landmarks)

            # Extract lips keypoints
            lips_indices = (
                self.LIPS_LANDMARKS['upper_lip'] +
                self.LIPS_LANDMARKS['lower_lip']
            )

            lips_keypoints_list = []
            for idx in lips_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    lips_keypoints_list.extend([lm.x, lm.y, lm.z])

                    # Optionally draw keypoints on frame
                    if self._draw_keypoints:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            if lips_keypoints_list:
                lips_keypoints = np.array(lips_keypoints_list)

            # Extract eye keypoints for normalization
            # Left eye corners: 33, 133, 157, 158
            left_eye_indices = [33, 133, 157, 158]
            left_eye_list = []
            for idx in left_eye_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    left_eye_list.extend([lm.x, lm.y, lm.z])

            if left_eye_list:
                left_eye = np.array(left_eye_list)

            # Right eye corners: 362, 263, 386, 387
            right_eye_indices = [362, 263, 386, 387]
            right_eye_list = []
            for idx in right_eye_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    right_eye_list.extend([lm.x, lm.y, lm.z])

            if right_eye_list:
                right_eye = np.array(right_eye_list)

        return {
            "frame": frame,
            "lips": lips_keypoints,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "head_pose": head_pose,
            "face_landmarks": face_landmarks_array
        }
