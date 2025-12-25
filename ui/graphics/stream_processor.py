"""
stream_processor.py
Main coordinator for video stream processing
"""

import cv2
import threading
import time

from ..managers.processor_manager import ProcessorManager
from ..managers.feature_manager import FeatureManager
from ..managers.frame_processor import FrameProcessor
from ..managers.camera_manager import CameraManager
from ..managers.ai_queue_manager import AIQueueManager


DEFAULT_AI_W = 210
DEFAULT_AI_H = 300
DEFAULT_FPS = 25.0


class StreamProcessor:
    """Main class for coordinating video stream processing"""
    
    def __init__(self, state,
                 camera_index: int = 0,
                 ai_w: int = DEFAULT_AI_W,
                 ai_h: int = DEFAULT_AI_H,
                 target_fps: float = DEFAULT_FPS,
                 ai_out_dir: str = "preprocessing/out",
                 ai_queue_max: int = 128):
        
        self.state = state
        self.TARGET_FPS = target_fps
        self.FRAME_INTERVAL = 1.0 / target_fps
        
        # Initialize managers
        self.processor_manager = ProcessorManager()
        self.feature_manager = FeatureManager(self.processor_manager.console)
        self.camera_manager = CameraManager(camera_index)
        self.ai_queue_manager = AIQueueManager(ai_queue_max)
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(
            self.processor_manager,
            self.feature_manager,
            state,
            ai_w,
            ai_h
        )
        
        # Runtime controls
        self._stop_event = threading.Event()
        self._thread = None
        self._preview_frame = None
    
    # ------------------- Public API for Feature Toggles -------------------
    
    def toggle_segmentation(self, enabled):
        self.feature_manager.toggle_segmentation(enabled)
    
    def toggle_hands(self, enabled):
        self.feature_manager.toggle_hands(enabled)
    
    def toggle_pose(self, enabled):
        self.feature_manager.toggle_pose(enabled)
    
    def toggle_clahe(self, enabled):
        self.feature_manager.toggle_clahe(enabled)
    
    def toggle_brightness(self, enabled):
        self.feature_manager.toggle_brightness(enabled)
    
    def toggle_crop(self, enabled):
        self.feature_manager.toggle_crop(enabled)
    
    def toggle_contour(self, enabled):
        self.feature_manager.toggle_contour(enabled)
    
    # ------------------- Public API for Processor Access -------------------
    
    def get_brightness_processor(self):
        return self.processor_manager.get_brightness_processor()
    
    def get_clahe_processor(self):
        return self.processor_manager.get_clahe_processor()
    
    def get_crop_processor(self):
        return self.processor_manager.get_crop_processor()
    
    def get_mask_combiner(self):
        return self.processor_manager.get_mask_combiner()
    
    def get_background_processor(self):
        return self.processor_manager.get_background_processor()
    
    def get_contour_drawer(self):
        return self.processor_manager.get_contour_drawer()
    
    def get_segmentation_processor(self):
        return self.processor_manager.get_segmentation_processor()
    
    def get_hand_processor(self):
        return self.processor_manager.get_hand_processor()
    
    def get_pose_processor(self):
        return self.processor_manager.get_pose_processor()
    
    # ------------------- Start / Stop -------------------
    
    def start(self, show_preview: bool = True):
        """Start the stream processing"""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Already running")
        
        self.camera_manager.start()
        self._stop_event.clear()
        
        self.ai_queue_manager.start_worker(self._stop_event)
        
        self._thread = threading.Thread(
            target=self._main_loop,
            args=(show_preview,),
            daemon=True
        )
        self._thread.start()
    
    def stop(self):
        """Stop the stream processing"""
        self._stop_event.set()
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        self.ai_queue_manager.stop_worker()
        self.camera_manager.stop()
        cv2.destroyAllWindows()
    
    def get_preview(self):
        """Get the latest preview frame"""
        return None if self._preview_frame is None else self._preview_frame.copy()
    
    # ------------------- Main Processing Loop -------------------
    
    def _main_loop(self, show_preview: bool):
        """Main loop that processes frames continuously"""
        frame_idx = 0
        next_push_time = time.time()
        
        while not self._stop_event.is_set():
            # Read frame from camera
            ok, frame = self.camera_manager.read_frame()
            if not ok:
                continue
            
            # Process frame
            ai_rgba, preview = self.frame_processor.process_frame(frame)
            
            # Push to AI queue with fixed rate
            now = time.time()
            if now < next_push_time:
                time.sleep(next_push_time - now)
                now = next_push_time
            
            self.ai_queue_manager.push_frame(ai_rgba)
            next_push_time += self.FRAME_INTERVAL
            frame_idx += 1
            
            # Update preview
            self._update_preview(preview, show_preview, ai_rgba)
    
    def _update_preview(self, preview, show_preview, ai_rgba):
        """Update preview frame and optionally show OpenCV windows"""
        try:
            preview_small = cv2.resize(preview, (960//2, 720//3), 
                                      interpolation=cv2.INTER_AREA)
        except Exception:
            preview_small = preview.copy()
        
        self._preview_frame = preview_small
        
        # Optional: Show OpenCV windows
        if show_preview:
            cv2.imshow("User preview", preview_small)
            view_final = self._create_final_view(ai_rgba)
            cv2.imshow("Final (AI crop over white)", view_final)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop_event.set()
    
    def _create_final_view(self, ai_rgba):
        """Create final view with white background"""
        import numpy as np
        view_final = np.where(
            (ai_rgba[:,:,3]==255)[:,:,None],
            ai_rgba[:,:,:3],
            np.full_like(ai_rgba[:,:,:3], 255)
        )
        return view_final
