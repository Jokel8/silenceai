"""
stream_processor.py

StreamProcessor with preprocessing toggle:
 - preprocessing_enabled: if True -> background removed + CLAHE+brightness applied.
   Preview: raw background + processed person foreground + green outline.
   AI: processed RGBA crops (transparent background).
 - if False:
   Preview: raw only.
   AI: raw resized -> BGRA (alpha=255).
"""

import cv2
import numpy as np
import mediapipe as mp
import threading, queue, os, time
from typing import Optional
import importlib.util

try:
    from preprocessing import Preprocessing as prepro
except ImportError:
    preprozessing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'userInterfaces'))
    spec = importlib.util.spec_from_file_location("stream_processor", os.path.join(preprozessing_dir, "preprocessing.py"))
    preprocessing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing_module)
    strproc = preprocessing_module.Preprocessing

# ------------------- Defaults / Config -------------------
DEFAULT_AI_W = 210
DEFAULT_AI_H = 300   # size, because got streched in trainingsdata
DEFAULT_FPS = 25.0

class StreamProcessor:
    def __init__(self, state,
                 camera_index:int = 0,
                 ai_w: int = DEFAULT_AI_W,
                 ai_h: int = DEFAULT_AI_H,
                 target_fps: float = DEFAULT_FPS,
                 ai_out_dir: str = "preprocessing/out",
                 ai_queue_max:int = 128,
                 seg_threshold:float = 0.4,
                 hand_conf:float = 0.5,
                 pose_conf:float = 0.5,
                 clahe_clip:float = 2.0,
                 clahe_tile=(8,8),
                 crop_padding:float = 1.08,
                 crop_min_frac:float = 0.42):
        self.camera_index = camera_index
        self.AI_W = ai_w
        self.AI_H = ai_h
        self.TARGET_FPS = target_fps
        self.FRAME_INTERVAL = 1.0 / target_fps
        self.ai_q = queue.Queue(maxsize=ai_queue_max)

        # MediaPipe params
        self.SEG_THRESHOLD = seg_threshold
        self.HAND_CONF = hand_conf
        self.POSE_CONF = pose_conf
        self.CLAHE_CLIP = clahe_clip
        self.CLAHE_TILE = clahe_tile
        self.CROP_PADDING = crop_padding
        self.CROP_MIN_FRAC = crop_min_frac

        # MediaPipe models (initialized on start)
        self.mp_selfie = None
        self.mp_hands = None
        self.mp_pose = None

        # runtime controls
        self._stop_event = threading.Event()
        self._thread = None
        self._ai_worker = None
        self._cap = None
        self._preview_frame = None
        self._brightness = 1.0

        self.state = state
    # ------------------- Schnittstelle pipline -------------------
    def _ai_saver_worker(self, stop_event):
        idx = 0
        while not stop_event.is_set() or not self.ai_q.empty():
            try:
                rgba = self.ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            ai_output = rgba.COLOR_BGRA2BGR
            yield ai_output
            idx += 1

    # ------------------- Start / Stop -------------------
    def start(self, show_preview: bool = True):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Already running")
        # init mediapipe
        self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                                 min_detection_confidence=self.HAND_CONF,
                                                 min_tracking_confidence=self.HAND_CONF)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                              min_detection_confidence=self.POSE_CONF,
                                              min_tracking_confidence=self.POSE_CONF)
        self._cap = cv2.VideoCapture(self.camera_index)
        self._stop_event.clear()
        # ai saver
        self._ai_worker = threading.Thread(target=self._ai_saver_worker, args=(self._stop_event,), daemon=True)
        self._ai_worker.start()
        # main loop thread
        self._thread = threading.Thread(target=self._main_loop, args=(show_preview,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ai_worker is not None:
            self._ai_worker.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    # ------------------- Public getter for preview -------------------
    def get_preview(self):
        return None if self._preview_frame is None else self._preview_frame.copy()

    # ------------------- Main processing loop -------------------
    def _main_loop(self, show_preview: bool):
        frame_idx = 0
        next_push_time = time.time()
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Selfie Seg
            seg_res = self.mp_selfie.process(frame_rgb)
            seg_mask = seg_res.segmentation_mask if seg_res and seg_res.segmentation_mask is not None else None
            if seg_mask is None:
                base_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                base_mask = (seg_mask > self.SEG_THRESHOLD).astype(np.uint8)

            # Hands
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            hands_res = self.mp_hands.process(frame_rgb)
            if hands_res.multi_hand_landmarks:
                for hand_landmarks in hands_res.multi_hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    if len(pts) >= 3:
                        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                        cv2.fillConvexPoly(hand_mask, hull, 1)

            # Torso / pose
            torso_mask = np.zeros((h, w), dtype=np.uint8)
            pose_res = self.mp_pose.process(frame_rgb)
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                indices = []
                for idx in [0,11,12,23,24]:
                    if idx < len(lm):
                        indices.append(lm[idx])
                if len(indices) >= 3:
                    torso_mask = prepro._landmarks_to_mask(prepro, indices, frame.shape)

            # combine + smooth
            combined = np.clip(base_mask + hand_mask + torso_mask, 0, 1).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            combined_255 = (combined * 255).astype(np.uint8)

            # Preprocess chain (CLAHE + brightness) applied to a copy of the frame
            clahe_img = prepro._apply_clahe(prepro, frame)
            bright_img = prepro._adjust_brightness(prepro, clahe_img, self._brightness)

            # ------------------ Decide AI frame & preview based on preprocessing flag ------------------
            if self.state.usePreProcessing:
                # AI receives processed RGBA (processed + mask alpha), as before
                ai_rgba, _coords = prepro._crop_and_resize_by_mask(prepro, bright_img, combined_255,
                                                                 target_w=self.AI_W, target_h=self.AI_H)
                # Preview: raw background, processed person foreground (compose), draw green contour
                raw_bg = frame.copy()
                proc_fg = bright_img.copy()
                alpha_f = (combined_255.astype(np.float32) / 255.0)[:,:,None]
                preview_comp = (proc_fg.astype(np.float32)*alpha_f + raw_bg.astype(np.float32)*(1-alpha_f)).astype(np.uint8)
                # draw green contour around person for animation effect
                contours, _ = cv2.findContours(combined_255.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(preview_comp, contours, -1, (0,255,0), thickness=3)
                    # subtle translucent thicker outline for animation base
                    overlay = preview_comp.copy()
                    cv2.drawContours(overlay, contours, -1, (0,255,0), thickness=8)
                    preview_comp = cv2.addWeighted(overlay, 0.22, preview_comp, 0.78, 0)
                preview = preview_comp
            else:
                # Preprocessing disabled: AI gets raw resized (BGRA alpha=255), preview shows raw only
                resized_raw = cv2.resize(frame, (self.AI_W, self.AI_H), interpolation=cv2.INTER_AREA)
                ai_rgba = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2BGRA)
                ai_rgba[:, :, 3] = 255
                preview = frame.copy()  # raw only

            # push AI frame at fixed rate (1/TARGET_FPS)
            now = time.time()
            if now < next_push_time:
                time.sleep(next_push_time - now)
                now = next_push_time
            try:
                self.ai_q.put_nowait(ai_rgba)
            except queue.Full:
                # drop if full
                pass
            next_push_time += self.FRAME_INTERVAL
            frame_idx += 1

            # set preview (resized to reasonable size for UI)
            try:
                preview_small = cv2.resize(preview, (960//2, 720//3), interpolation=cv2.INTER_AREA)  # ~320x240
            except Exception:
                preview_small = preview.copy()
            self._preview_frame = preview_small

            if show_preview:
                # show single preview window (no mask split)
                cv2.imshow("User preview", preview_small)
                # quick AI view
                view_final = np.where((ai_rgba[:,:,3]==255)[:,:,None], ai_rgba[:,:,:3], np.full_like(ai_rgba[:,:,:3], 255))
                cv2.imshow("Final (AI crop over white)", view_final)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop_event.set()
                    break
