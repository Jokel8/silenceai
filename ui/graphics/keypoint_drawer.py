"""
keypoint_drawer_processor.py
Handles visualization of hand keypoints on frames
Draws keypoints as light-blue squares and connections between fingers
"""

import cv2
import numpy as np


class KeypointDrawerProcessor:
    """Draws hand keypoints and skeleton connections on frames"""
    
    # MediaPipe hand landmark connections
    # These connections define the hand skeleton structure
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]
    
    def __init__(self, keypoint_size=5, line_thickness=2, color=(255, 255, 0)):
        """
        Initialize the drawer
        
        Args:
            keypoint_size: Size of the square markers (half-width in pixels)
            line_thickness: Thickness of connection lines
            color: BGR color for drawing (default: light cyan)
        """
        self.keypoint_size = keypoint_size
        self.line_thickness = line_thickness
        self.color = color  # BGR format: (255, 255, 0) = cyan/light blue
        self.lips_color = (0, 0, 255)  # BGR format: (0, 0, 255) = red
        self.face_color = (0, 0, 255)  # BGR format: (0, 0, 255) = red for face pose

    def set_color(self, color):
        """Set drawing color in BGR format"""
        self.color = color

    def set_lips_color(self, color):
        """Set lips drawing color in BGR format"""
        self.lips_color = color

    def set_face_color(self, color):
        """Set face pose drawing color in BGR format"""
        self.face_color = color

    def _draw_keypoints_and_connections(self, frame, keypoints_array, frame_height, frame_width):
        """
        Draw keypoints and connections for one hand
        
        Args:
            frame: Image to draw on
            keypoints_array: Array of 63 values (21 keypoints × 3 coordinates)
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
        """
        # Skip if no keypoints detected (all zeros)
        if np.all(keypoints_array == 0):
            return

        # Reshape to get individual keypoints (21 keypoints × 3 coordinates)
        keypoints = keypoints_array.reshape(21, 3)
        
        # Convert normalized coordinates to pixel positions
        pixel_coords = []
        for kp in keypoints:
            x = int(kp[0] * frame_width)
            y = int(kp[1] * frame_height)
            pixel_coords.append((x, y))
        
        # Draw connections (skeleton)
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                pt1 = pixel_coords[start_idx]
                pt2 = pixel_coords[end_idx]
                cv2.line(frame, pt1, pt2, self.color, self.line_thickness)
        
        # Draw keypoints as squares
        for x, y in pixel_coords:
            # Draw square centered at (x, y)
            top_left = (x - self.keypoint_size, y - self.keypoint_size)
            bottom_right = (x + self.keypoint_size, y + self.keypoint_size)
            cv2.rectangle(frame, top_left, bottom_right, self.color, -1)

    def _draw_lips_keypoints(self, frame, lips_array, frame_height, frame_width):
        """
        Draw lips keypoints and connections for one face
        
        Args:
            frame: Image to draw on
            lips_array: Array of 54 values (18 keypoints × 3 coordinates)
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
        """
        # Skip if no keypoints detected (all zeros)
        if np.all(lips_array == 0):
            return

        # Reshape to get individual keypoints (18 keypoints × 3 coordinates)
        lips_pts = lips_array.reshape(18, 3)
        
        # Convert normalized coordinates to pixel positions
        pixel_coords = []
        for kp in lips_pts:
            x = int(kp[0] * frame_width)
            y = int(kp[1] * frame_height)
            pixel_coords.append((x, y))
        
        # Draw connections for upper lip (indices 0-8)
        for i in range(8):
            pt1 = pixel_coords[i]
            pt2 = pixel_coords[i + 1]
            cv2.line(frame, pt1, pt2, self.lips_color, self.line_thickness)
        
        # Draw connections for lower lip (indices 9-17)
        for i in range(9, 17):
            pt1 = pixel_coords[i]
            pt2 = pixel_coords[i + 1]
            cv2.line(frame, pt1, pt2, self.lips_color, self.line_thickness)
        
        # Connect mouth corners (index 0 to 17 for closure)
        cv2.line(frame, pixel_coords[0], pixel_coords[17], self.lips_color, self.line_thickness)
        cv2.line(frame, pixel_coords[8], pixel_coords[9], self.lips_color, self.line_thickness)
        
        # Draw keypoints as circles with red color
        for x, y in pixel_coords:
            cv2.circle(frame, (x, y), self.keypoint_size, self.lips_color, -1)

    def _draw_focus_bracket(self, frame, x_min, y_min, x_max, y_max, color, thickness=2, corner_length=20):
        """
        Draw a focus bracket (like camera autofocus box) with rounded corners
        The corners are separated in the middle of each edge
        
        Args:
            frame: Image to draw on
            x_min, y_min: Top-left corner
            x_max, y_max: Bottom-right corner
            color: BGR color
            thickness: Line thickness
            corner_length: Length of each corner segment
        """
        # Top-left corner
        cv2.line(frame, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)

    def _draw_head_pose_3d(self, frame, face_landmarks_array, head_pose, frame_height, frame_width):
        """
        Draw 3D head pose visualization with bounding box and axis from nose
        
        Args:
            frame: Image to draw on
            face_landmarks_array: Array of all 468 face landmarks (468 × 3)
            head_pose: Dict with 'pitch', 'yaw', 'roll' in degrees
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
        """
        try:
            # Check if array is empty or too small
            if face_landmarks_array is None or len(face_landmarks_array) == 0:
                return
            
            if np.all(face_landmarks_array == 0):
                return

            # Expected size: 468 * 3 = 1404
            if len(face_landmarks_array) < 1404:
                # Array too small, skip
                print("Array to small for face landmarks drawing")
                return

            num_landmarks = len(face_landmarks_array) // 3
            
            # Reshape landmarks (N landmarks × 3 coordinates)
            landmarks = face_landmarks_array.reshape(num_landmarks, 3)

            # Get key points for bounding box
            # Use face contour points (indices 0-16 are face outline)
            face_outline_indices = list(range(0, 17))  # Left to right face contour
            
            face_pts = []
            for idx in face_outline_indices:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    x = int(pt[0] * frame_width)
                    y = int(pt[1] * frame_height)
                    # Clamp to frame bounds
                    x = max(0, min(x, frame_width - 1))
                    y = max(0, min(y, frame_height - 1))
                    face_pts.append((x, y))

            if len(face_pts) > 2:
                # Draw bounding box around face
                x_coords = [p[0] for p in face_pts]
                y_coords = [p[1] for p in face_pts]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding to move the bounding box outside the face
                padding = 30
                x_min = max(0, x_min - padding)
                x_max = min(frame_width - 1, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(frame_height - 1, y_max + padding)

                # Draw bounding box as focus bracket (camera autofocus style)
                self._draw_focus_bracket(frame, x_min, y_min, x_max, y_max, self.face_color, thickness=3, corner_length=40)

            # Draw 3D axes from nose
            # Try using landmark index 168 (upper nose bridge) which may be more accurate
            # Fallback to index 1 (nose tip) if not available
            nose_idx = 4 if len(landmarks) > 3 else 1
            if len(landmarks) > nose_idx:
                nose = landmarks[nose_idx]
                nose_x = int(nose[0] * frame_width)
                nose_y = int(nose[1] * frame_height)
                nose_z = nose[2]
                
                # Clamp nose position
                nose_x = max(0, min(nose_x, frame_width - 1))
                nose_y = max(0, min(nose_y, frame_height - 1))

                # Axis length
                axis_length = 50

                # Calculate rotation angles from head pose
                # Apply pitch offset of -40° to correct calibration
                pitch_offset = 40.0
                pitch_rad = np.radians(head_pose.get('pitch', 0.0) + pitch_offset)
                yaw_rad = np.radians(head_pose.get('yaw', 0.0))

                # Create a pointer/direction vector from head pose (yaw and pitch)
                # This represents where the head is looking
                pointer_length = 100
                
                # Calculate direction based on yaw (horizontal) and pitch (vertical)
                pointer_x = -pointer_length * np.cos(pitch_rad) * np.sin(yaw_rad)
                pointer_y = -pointer_length * np.sin(pitch_rad)

                # Calculate endpoint with scaling factor for visibility
                pointer_2d_x = nose_x + int(pointer_x * 0.8)
                pointer_2d_y = nose_y + int(pointer_y * 0.8)

                # Clamp endpoint position
                pointer_2d_x = max(0, min(pointer_2d_x, frame_width - 1))
                pointer_2d_y = max(0, min(pointer_2d_y, frame_height - 1))

                # Draw main pointer line in cyan
                cv2.line(frame, (nose_x, nose_y), (pointer_2d_x, pointer_2d_y), self.face_color, 4)

                # Draw nose point
                cv2.circle(frame, (nose_x, nose_y), 6, self.face_color, -1)

        except Exception as e:
            print(f"Error drawing head pose: {e}")
            # Silent fail - just don't draw if something goes wrong
            pass

    def process(self, frame, keypoints_data):
        """
        Draw hand, lips, and face pose keypoints on frame
        
        Args:
            frame: BGR image frame to draw on
            keypoints_data: Dict with optional keys:
                           - 'left_hand': numpy array of 63 values
                           - 'right_hand': numpy array of 63 values
                           - 'lips': numpy array of 54 values
                           - 'face_landmarks': numpy array of 468*3 values
                           - 'head_pose': dict with pitch, yaw, roll
        
        Returns:
            Annotated frame with drawn keypoints
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Draw left hand keypoints
        if "left_hand" in keypoints_data:
            self._draw_keypoints_and_connections(
                frame,
                keypoints_data["left_hand"],
                frame_height,
                frame_width
            )
        
        # Draw right hand keypoints
        if "right_hand" in keypoints_data:
            self._draw_keypoints_and_connections(
                frame,
                keypoints_data["right_hand"],
                frame_height,
                frame_width
            )
        
        # Draw lips keypoints
        if "lips" in keypoints_data:
            self._draw_lips_keypoints(
                frame,
                keypoints_data["lips"],
                frame_height,
                frame_width
            )
        
        # Draw face pose (head orientation with 3D axes)
        if "face_landmarks" in keypoints_data and "head_pose" in keypoints_data:
            self._draw_head_pose_3d(
                frame,
                keypoints_data["face_landmarks"],
                keypoints_data["head_pose"],
                frame_height,
                frame_width
            )
        
        return frame
