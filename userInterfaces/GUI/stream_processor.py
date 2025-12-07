"""
stream_processor.py (Refactored)
Koordiniert alle Prozessoren für die Stream-Verarbeitung
"""

import cv2
import numpy as np
import threading
import queue
import time

# Import aller Prozessor-Klassen
from brightness_processor import BrightnessProcessor
from clahe_processor import CLAHEProcessor
from crop_resize_processor import CropResizeProcessor
from mask_combiner_processor import MaskCombinerProcessor
from background_removal_processor import BackgroundRemovalProcessor
from contour_drawer_processor import ContourDrawerProcessor
from segmentation_processor import SegmentationProcessor
from hand_detection_processor import HandDetectionProcessor
from pose_detection_processor import PoseDetectionProcessor


DEFAULT_AI_W = 210
DEFAULT_AI_H = 300
DEFAULT_FPS = 25.0


class StreamProcessor:
    """Hauptklasse zur Koordination aller Bildverarbeitungs-Prozessoren"""
    
    def __init__(self, state,
                 camera_index: int = 0,
                 ai_w: int = DEFAULT_AI_W,
                 ai_h: int = DEFAULT_AI_H,
                 target_fps: float = DEFAULT_FPS,
                 ai_out_dir: str = "preprocessing/out",
                 ai_queue_max: int = 128):
        
        self.camera_index = camera_index
        self.AI_W = ai_w
        self.AI_H = ai_h
        self.TARGET_FPS = target_fps
        self.FRAME_INTERVAL = 1.0 / target_fps
        self.ai_q = queue.Queue(maxsize=ai_queue_max)
        self.state = state
        
        # Initialisiere alle Prozessoren
        self.brightness_proc = BrightnessProcessor(initial_factor=1.0)
        self.clahe_proc = CLAHEProcessor(clip_limit=2.0, tile_grid_size=(8, 8))
        self.crop_proc = CropResizeProcessor(padding=1.08, min_frac=0.42)
        self.mask_combiner = MaskCombinerProcessor(kernel_size=(7, 7))
        self.background_proc = BackgroundRemovalProcessor()
        self.contour_drawer = ContourDrawerProcessor(
            color=(0, 255, 0), 
            thickness=3,
            overlay_thickness=8,
            overlay_alpha=0.22
        )
        self.segmentation_proc = SegmentationProcessor(threshold=0.4)
        self.hand_proc = HandDetectionProcessor(confidence=0.5)
        self.pose_proc = PoseDetectionProcessor(confidence=0.5)
        
        # Runtime controls
        self._stop_event = threading.Event()
        self._thread = None
        self._ai_worker = None
        self._cap = None
        self._preview_frame = None
    
    # ------------------- Öffentliche Getter/Setter für Prozessor-Steuerung -------------------
    
    def get_brightness_processor(self):
        """Zugriff auf Brightness Processor"""
        return self.brightness_proc
    
    def get_clahe_processor(self):
        """Zugriff auf CLAHE Processor"""
        return self.clahe_proc
    
    def get_crop_processor(self):
        """Zugriff auf Crop/Resize Processor"""
        return self.crop_proc
    
    def get_mask_combiner(self):
        """Zugriff auf Mask Combiner"""
        return self.mask_combiner
    
    def get_background_processor(self):
        """Zugriff auf Background Removal Processor"""
        return self.background_proc
    
    def get_contour_drawer(self):
        """Zugriff auf Contour Drawer"""
        return self.contour_drawer
    
    def get_segmentation_processor(self):
        """Zugriff auf Segmentation Processor"""
        return self.segmentation_proc
    
    def get_hand_processor(self):
        """Zugriff auf Hand Detection Processor"""
        return self.hand_proc
    
    def get_pose_processor(self):
        """Zugriff auf Pose Detection Processor"""
        return self.pose_proc
    
    # ------------------- AI Worker Thread -------------------
    
    def _ai_saver_worker(self, stop_event):
        """Worker thread für AI-Frame-Verarbeitung"""
        idx = 0
        while not stop_event.is_set() or not self.ai_q.empty():
            try:
                rgba = self.ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            ai_output = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
            yield ai_output
            idx += 1
    
    # ------------------- Start / Stop -------------------
    
    def start(self, show_preview: bool = True):
        """Starte Stream-Verarbeitung"""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Already running")
        
        self._cap = cv2.VideoCapture(self.camera_index)
        self._stop_event.clear()
        
        # AI saver thread
        self._ai_worker = threading.Thread(
            target=self._ai_saver_worker,
            args=(self._stop_event,),
            daemon=True
        )
        self._ai_worker.start()
        
        # Main loop thread
        self._thread = threading.Thread(
            target=self._main_loop,
            args=(show_preview,),
            daemon=True
        )
        self._thread.start()
    
    def stop(self):
        """Stoppe Stream-Verarbeitung"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ai_worker is not None:
            self._ai_worker.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()
    
    def get_preview(self):
        """Hole aktuelles Preview-Frame"""
        return None if self._preview_frame is None else self._preview_frame.copy()
    
    # ------------------- Haupt-Verarbeitungsloop -------------------
    
    def _main_loop(self, show_preview: bool):
        """Hauptloop für Frame-Verarbeitung"""
        frame_idx = 0
        next_push_time = time.time()
        
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Segmentierung mit allen Prozessoren
            base_mask = self.segmentation_proc.process(frame_rgb)
            hand_mask = self.hand_proc.process(frame_rgb)
            torso_mask = self.pose_proc.process(frame_rgb)
            
            # 2. Masken kombinieren und glätten
            combined_255 = self.mask_combiner.process(base_mask, hand_mask, torso_mask)
            
            # 3. Bildverbesserungen anwenden
            clahe_img = self.clahe_proc.process(frame)
            bright_img = self.brightness_proc.process(clahe_img)
            
            # 4. Entscheide basierend auf Preprocessing-Flag
            if self.state.usePreProcessing:
                # AI: Verarbeitetes RGBA
                ai_rgba, _coords = self.crop_proc.process(
                    bright_img, combined_255, self.AI_W, self.AI_H
                )
                
                # Preview: Roher Hintergrund + verarbeiteter Vordergrund + Kontur
                raw_bg = frame.copy()
                preview_comp = self.background_proc.composite_on_background(
                    bright_img, combined_255, raw_bg
                )
                preview = self.contour_drawer.process(preview_comp, combined_255)
            else:
                # Preprocessing deaktiviert: Rohes Bild
                resized_raw = cv2.resize(frame, (self.AI_W, self.AI_H), interpolation=cv2.INTER_AREA)
                ai_rgba = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2BGRA)
                ai_rgba[:, :, 3] = 255
                preview = frame.copy()
            
            # 5. AI-Frame mit fester Rate pushen
            now = time.time()
            if now < next_push_time:
                time.sleep(next_push_time - now)
                now = next_push_time
            
            try:
                self.ai_q.put_nowait(ai_rgba)
            except queue.Full:
                pass  # Frame droppen wenn Queue voll
            
            next_push_time += self.FRAME_INTERVAL
            frame_idx += 1
            
            # 6. Preview aktualisieren
            try:
                preview_small = cv2.resize(preview, (960//2, 720//3), interpolation=cv2.INTER_AREA)
            except Exception:
                preview_small = preview.copy()
            
            self._preview_frame = preview_small
            
            # 7. Optional: OpenCV-Fenster anzeigen
            if show_preview:
                cv2.imshow("User preview", preview_small)
                view_final = np.where(
                    (ai_rgba[:,:,3]==255)[:,:,None],
                    ai_rgba[:,:,:3],
                    np.full_like(ai_rgba[:,:,:3], 255)
                )
                cv2.imshow("Final (AI crop over white)", view_final)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop_event.set()
                    break
