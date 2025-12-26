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
            ai_rgba, crop_coords = self._create_ai_output_with_preprocessing(processed_img, combined_255)
            preview = self._create_preview_with_preprocessing(frame, processed_img, combined_255, crop_coords)
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
            ai_rgba, coords = self.pm.crop_proc.process(
                processed_img, combined_255, self.AI_W, self.AI_H
            )
        else:
            # Without crop: Simply resize
            resized = cv2.resize(processed_img, (self.AI_W, self.AI_H), 
                               interpolation=cv2.INTER_AREA)
            ai_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
            ai_rgba[:, :, 3] = cv2.resize(combined_255, (self.AI_W, self.AI_H), 
                                         interpolation=cv2.INTER_NEAREST)
            return ai_rgba, None

        return ai_rgba, coords
    
    def _create_ai_output_without_preprocessing(self, frame):
        """Create AI output without preprocessing"""
        resized_raw = cv2.resize(frame, (self.AI_W, self.AI_H), 
                                interpolation=cv2.INTER_AREA)
        ai_rgba = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2BGRA)
        ai_rgba[:, :, 3] = 255
        
        return ai_rgba
    
    def _create_preview_with_preprocessing(self, raw_frame, processed_img, combined_255, crop_coords=None):
        """Create preview with preprocessing enabled"""
        h, w = raw_frame.shape[:2]
        # If crop is enabled, prefer showing the zoomed/cropped region as preview
        if self.fm.use_crop and crop_coords is not None:
            try:
                x1 = int(crop_coords.get('x1', 0))
                y1 = int(crop_coords.get('y1', 0))
                x2 = int(crop_coords.get('x2', raw_frame.shape[1]))
                y2 = int(crop_coords.get('y2', raw_frame.shape[0]))

                # clip coords to image bounds
                x1 = max(0, min(x1, raw_frame.shape[1] - 1))
                x2 = max(0, min(x2, raw_frame.shape[1]))
                y1 = max(0, min(y1, raw_frame.shape[0] - 1))
                y2 = max(0, min(y2, raw_frame.shape[0]))

                if x2 > x1 and y2 > y1:
                    # crop processed image, mask and raw frame to the detected bbox
                    crop_processed = processed_img[y1:y2, x1:x2]
                    crop_mask = combined_255[y1:y2, x1:x2]
                    crop_raw = raw_frame[y1:y2, x1:x2]

                    # Composite on the cropped region
                    preview_small = self.pm.background_proc.composite_on_background(
                        crop_processed, crop_mask, crop_raw
                    )

                    # If hands detection is enabled, extract keypoints from the cropped processed image
                    kp_data = None
                    if self.fm.use_hands:
                        try:
                            kp_data = self.pm.hand_proc.extractKeypoints(crop_processed)
                        except Exception:
                            kp_data = None

                    # Draw keypoints on the small preview if available
                    if kp_data is not None:
                        try:
                            preview_small = self.keypoint_drawer.process(preview_small, kp_data)
                        except Exception:
                            pass

                    # Draw contour on the small preview using the cropped mask
                    if self.fm.use_contour:
                        try:
                            preview_small = self.pm.contour_drawer.process(preview_small, crop_mask)
                        except Exception:
                            pass

                    # Resize cropped preview to full preview area
                    try:
                        preview_cropped = cv2.resize(preview_small, (w, h), interpolation=cv2.INTER_LINEAR)
                        # store last crop in state for other components if needed
                        try:
                            setattr(self.state, 'last_crop_coords', crop_coords)
                            setattr(self.state, 'last_cropped_preview', preview_cropped)
                        except Exception:
                            pass
                        preview_comp = preview_cropped
                    except Exception:
                        # fallback to composite of full frame if resize fails
                        preview_comp = self.pm.background_proc.composite_on_background(
                            processed_img, combined_255, raw_frame
                        )
                else:
                    preview_comp = self.pm.background_proc.composite_on_background(
                        processed_img, combined_255, raw_frame
                    )
            except Exception:
                preview_comp = self.pm.background_proc.composite_on_background(
                    processed_img, combined_255, raw_frame
                )
        else:
            # No cropping: full composite preview
            preview_comp = self.pm.background_proc.composite_on_background(
                processed_img, combined_255, raw_frame
            )
        # If hands detection is enabled, extract keypoints (unless already extracted for cropped flow),
        # normalize and analyze
        try:
            if self.fm.use_hands:
                if 'kp_data' not in locals() or kp_data is None:
                    try:
                        kp_data = self.pm.hand_proc.extractKeypoints(processed_img)
                    except Exception:
                        kp_data = None

                # Draw keypoints on preview only when showing full composite (not on cropped preview)
                try:
                    if not (self.fm.use_crop and crop_coords is not None) and kp_data is not None:
                        preview_comp = self.keypoint_drawer.process(preview_comp, kp_data)
                except Exception:
                    pass

                # Normalize keypoints for analysis
                try:
                    normalized = self.keypoint_normalizer.process({
                        'left_hand': kp_data.get('left_hand', np.zeros(63)) if kp_data else np.zeros(63),
                        'right_hand': kp_data.get('right_hand', np.zeros(63)) if kp_data else np.zeros(63)
                    })
                except Exception:
                    normalized = {'left_hand': np.zeros(63), 'right_hand': np.zeros(63)}

                # Analyze gestures and store top-3 guesses in state
                try:
                    result = self.gesture_analyzer.analyze(normalized)
                    try:
                        top3 = result.get('top_3', [])
                        guesses = []
                        for label, conf in top3:
                            percent = float(conf) * 100.0
                            guesses.append((str(label), percent))

                        if hasattr(self.state, 'gesture_guesses'):
                            self.state.gesture_guesses = guesses
                        else:
                            setattr(self.state, 'gesture_guesses', guesses)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        # Draw contour only for the non-cropped preview here (cropped flow already applied contour)
        if self.fm.use_contour:
            if self.fm.use_crop and crop_coords is not None:
                return preview_comp
            else:
                return self.pm.contour_drawer.process(preview_comp, combined_255)
        else:
            return preview_comp
