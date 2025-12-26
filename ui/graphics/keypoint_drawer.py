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

    def set_color(self, color):
        """Set drawing color in BGR format"""
        self.color = color

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

    def process(self, frame, keypoints_data):
        """
        Draw hand keypoints on frame
        
        Args:
            frame: BGR image frame to draw on
            keypoints_data: Dict with keys 'left_hand' and 'right_hand'
                           Each contains numpy array of 63 values
        
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
        
        return frame
