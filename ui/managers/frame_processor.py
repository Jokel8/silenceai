"""
frame_processor.py
Handles individual frame processing logic
"""

import cv2
import numpy as np

from ui.graphics.keypoint_drawer import KeypointDrawerProcessor
from preprocessing.keypoint_normalization_processor import KeypointsNormalizeProcessor
from coreprocessing.gesture_analyzer import GestureAnalyzer


class FrameProcessor:
    """Processes individual frames through the pipeline"""
    
    def __init__(self, processor_manager, feature_manager, state, ai_w, ai_h):
        self.pm = processor_manager
        self.fm = feature_manager
        self.state = state
        self.AI_W = ai_w
        self.AI_H = ai_h
        # Initialize processors used in GUI pipeline
        self.keypoint_drawer = KeypointDrawerProcessor(keypoint_size=4, line_thickness=2, color=(255, 255, 0))
        self.keypoint_normalizer = KeypointsNormalizeProcessor()
        self.gesture_analyzer = GestureAnalyzer()
    
    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline
        
        Returns:
            tuple: (ai_rgba, preview_frame)
        """
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Create masks (only if enabled)
        masks = self._create_masks(frame_rgb)
        
        # 2. Combine masks
        combined_255 = self._combine_masks(masks, h, w)
        
        # 3. Apply image enhancements
        processed_img = self._apply_enhancements(frame)
        
        # 4. Create AI output and preview based on preprocessing flag
        if self.state.usePreProcessing:
            ai_rgba = self._create_ai_output_with_preprocessing(processed_img, combined_255)
            preview = self._create_preview_with_preprocessing(frame, processed_img, combined_255)
        else:
            ai_rgba = self._create_ai_output_without_preprocessing(frame)
            preview = frame.copy()
        
        return ai_rgba, preview
    
    def _create_masks(self, frame_rgb):
        """Create masks from enabled detection methods"""
        masks = []
        
        if self.fm.use_segmentation:
            masks.append(self.pm.segmentation_proc.process(frame_rgb))
        
        if self.fm.use_hands:
            masks.append(self.pm.hand_proc.process(frame_rgb))
        
        if self.fm.use_pose:
            masks.append(self.pm.pose_proc.process(frame_rgb))
        
        return masks
    
    def _combine_masks(self, masks, h, w):
        """Combine multiple masks into one"""
        if masks:
            return self.pm.mask_combiner.process(*masks)
        else:
            return np.ones((h, w), dtype=np.uint8) * 255
    
    def _apply_enhancements(self, frame):
        """Apply image enhancements if enabled"""
        processed_img = frame.copy()
        
        if self.fm.use_clahe:
            processed_img = self.pm.clahe_proc.process(processed_img)
        
        if self.fm.use_brightness:
            processed_img = self.pm.brightness_proc.process(processed_img)
        
        return processed_img
    
    def _create_ai_output_with_preprocessing(self, processed_img, combined_255):
        """Create AI output with preprocessing enabled"""
        if self.fm.use_crop:
            ai_rgba, _coords = self.pm.crop_proc.process(
                processed_img, combined_255, self.AI_W, self.AI_H
            )
        else:
            # Without crop: Simply resize
            resized = cv2.resize(processed_img, (self.AI_W, self.AI_H), 
                               interpolation=cv2.INTER_AREA)
            ai_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
            ai_rgba[:, :, 3] = cv2.resize(combined_255, (self.AI_W, self.AI_H), 
                                         interpolation=cv2.INTER_NEAREST)
        
        return ai_rgba
    
    def _create_ai_output_without_preprocessing(self, frame):
        """Create AI output without preprocessing"""
        resized_raw = cv2.resize(frame, (self.AI_W, self.AI_H), 
                                interpolation=cv2.INTER_AREA)
        ai_rgba = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2BGRA)
        ai_rgba[:, :, 3] = 255
        
        return ai_rgba
    
    def _create_preview_with_preprocessing(self, raw_frame, processed_img, combined_255):
        """Create preview with preprocessing enabled"""
        # Composite: Raw background + processed foreground
        preview_comp = self.pm.background_proc.composite_on_background(
            processed_img, combined_255, raw_frame
        )
        # If hands detection is enabled, extract keypoints, normalize and analyze
        try:
            if self.fm.use_hands:
                # Extract keypoints (returns dict with 'frame','left_hand','right_hand')
                kp_data = self.pm.hand_proc.extractKeypoints(processed_img)

                # Draw keypoints on preview
                try:
                    preview_comp = self.keypoint_drawer.process(preview_comp, kp_data)
                except Exception:
                    pass

                # Normalize keypoints for analysis
                try:
                    normalized = self.keypoint_normalizer.process({
                        'left_hand': kp_data.get('left_hand', np.zeros(63)),
                        'right_hand': kp_data.get('right_hand', np.zeros(63))
                    })
                except Exception:
                    normalized = {'left_hand': np.zeros(63), 'right_hand': np.zeros(63)}

                # Analyze gestures and overlay prediction
                try:
                    result = self.gesture_analyzer.analyze(normalized)
                    text = result.get('prediction', {}).get('text', '')
                    if text:
                        cv2.putText(preview_comp, str(text), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                except Exception:
                    # Analyzer might fail if model missing — ignore to keep GUI stable
                    pass
        except Exception:
            # Keep preview even if extraction or drawing fails
            pass

        # Draw contour if enabled
        if self.fm.use_contour:
            return self.pm.contour_drawer.process(preview_comp, combined_255)
        else:
            return preview_comp
