"""
graficInterface.py (Kombiniert)
Hauptdatei - Enthält StreamProcessor und UI in einer Datei
"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import ObjectProperty, NumericProperty
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, RoundedRectangle, Line
import os
import cv2
import numpy as np
import threading
import queue
import time

# Import aller Prozessor-Klassen
try:
    from userInterfaces.brightness_processor import BrightnessProcessor
    from userInterfaces.clahe_processor import CLAHEProcessor
    from userInterfaces.crop_resize_processor import CropResizeProcessor
    from userInterfaces.mask_combiner_processor import MaskCombinerProcessor
    from userInterfaces.background_removal_processor import BackgroundRemovalProcessor
    from userInterfaces.contour_drawer_processor import ContourDrawerProcessor
    from userInterfaces.segmentation_processor import SegmentationProcessor
    from userInterfaces.hand_detection_processor import HandDetectionProcessor
    from userInterfaces.pose_detection_processor import PoseDetectionProcessor
    from userInterfaces import consoleInterface
    Builder.load_file('userInterfaces/uiDesign.kv')
except ImportError:
    from brightness_processor import BrightnessProcessor
    from clahe_processor import CLAHEProcessor
    from crop_resize_processor import CropResizeProcessor
    from mask_combiner_processor import MaskCombinerProcessor
    from background_removal_processor import BackgroundRemovalProcessor
    from contour_drawer_processor import ContourDrawerProcessor
    from segmentation_processor import SegmentationProcessor
    from hand_detection_processor import HandDetectionProcessor
    from pose_detection_processor import PoseDetectionProcessor
    import consoleInterface
    Builder.load_file('uiDesign.kv')


DEFAULT_AI_W = 210
DEFAULT_AI_H = 300
DEFAULT_FPS = 25.0


# ==================== STREAM PROCESSOR ====================

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
        
        # Feature toggles (einzeln steuerbar)
        self.use_segmentation = True
        self.use_hands = True
        self.use_pose = True
        self.use_clahe = True
        self.use_brightness = True
        self.use_crop = True
        self.use_contour = True
        
        # Runtime controls
        self._stop_event = threading.Event()
        self._thread = None
        self._ai_worker = None
        self._cap = None
        self._preview_frame = None
    
    # ------------------- Öffentliche Getter für Prozessoren -------------------
    
    def get_brightness_processor(self):
        return self.brightness_proc
    
    def get_clahe_processor(self):
        return self.clahe_proc
    
    def get_crop_processor(self):
        return self.crop_proc
    
    def get_mask_combiner(self):
        return self.mask_combiner
    
    def get_background_processor(self):
        return self.background_proc
    
    def get_contour_drawer(self):
        return self.contour_drawer
    
    def get_segmentation_processor(self):
        return self.segmentation_proc
    
    def get_hand_processor(self):
        return self.hand_proc
    
    def get_pose_processor(self):
        return self.pose_proc
    
    # ------------------- Feature Toggle Methods -------------------
    
    def toggle_segmentation(self, enabled):
        self.use_segmentation = enabled
        consoleInterface.print_status(f"Segmentation: {'ON' if enabled else 'OFF'}")
    
    def toggle_hands(self, enabled):
        self.use_hands = enabled
        consoleInterface.print_status(f"Hand Detection: {'ON' if enabled else 'OFF'}")
    
    def toggle_pose(self, enabled):
        self.use_pose = enabled
        consoleInterface.print_status(f"Pose Detection: {'ON' if enabled else 'OFF'}")
    
    def toggle_clahe(self, enabled):
        self.use_clahe = enabled
        consoleInterface.print_status(f"CLAHE: {'ON' if enabled else 'OFF'}")
    
    def toggle_brightness(self, enabled):
        self.use_brightness = enabled
        consoleInterface.print_status(f"Brightness Adjustment: {'ON' if enabled else 'OFF'}")
    
    def toggle_crop(self, enabled):
        self.use_crop = enabled
        consoleInterface.print_status(f"Smart Crop: {'ON' if enabled else 'OFF'}")
    
    def toggle_contour(self, enabled):
        self.use_contour = enabled
        consoleInterface.print_status(f"Contour Drawing: {'ON' if enabled else 'OFF'}")
    
    # ------------------- AI Worker Thread -------------------
    
    def _ai_saver_worker(self, stop_event):
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
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Already running")
        
        self._cap = cv2.VideoCapture(self.camera_index)
        self._stop_event.clear()
        
        self._ai_worker = threading.Thread(
            target=self._ai_saver_worker,
            args=(self._stop_event,),
            daemon=True
        )
        self._ai_worker.start()
        
        self._thread = threading.Thread(
            target=self._main_loop,
            args=(show_preview,),
            daemon=True
        )
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
    
    def get_preview(self):
        return None if self._preview_frame is None else self._preview_frame.copy()
    
    # ------------------- Haupt-Verarbeitungsloop -------------------
    
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
            
            # 1. Masken erstellen (nur wenn aktiviert)
            masks = []
            if self.use_segmentation:
                masks.append(self.segmentation_proc.process(frame_rgb))
            if self.use_hands:
                masks.append(self.hand_proc.process(frame_rgb))
            if self.use_pose:
                masks.append(self.pose_proc.process(frame_rgb))
            
            # 2. Masken kombinieren
            if masks:
                combined_255 = self.mask_combiner.process(*masks)
            else:
                combined_255 = np.ones((h, w), dtype=np.uint8) * 255
            
            # 3. Bildverbesserungen anwenden (nur wenn aktiviert)
            processed_img = frame.copy()
            if self.use_clahe:
                processed_img = self.clahe_proc.process(processed_img)
            if self.use_brightness:
                processed_img = self.brightness_proc.process(processed_img)
            
            # 4. Entscheide basierend auf Preprocessing-Flag
            if self.state.usePreProcessing:
                # AI: Verarbeitetes RGBA mit oder ohne Crop
                if self.use_crop:
                    ai_rgba, _coords = self.crop_proc.process(
                        processed_img, combined_255, self.AI_W, self.AI_H
                    )
                else:
                    # Ohne Crop: Einfach resizen
                    resized = cv2.resize(processed_img, (self.AI_W, self.AI_H), interpolation=cv2.INTER_AREA)
                    ai_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
                    ai_rgba[:, :, 3] = cv2.resize(combined_255, (self.AI_W, self.AI_H), interpolation=cv2.INTER_NEAREST)
                
                # Preview: Roher Hintergrund + verarbeiteter Vordergrund
                raw_bg = frame.copy()
                preview_comp = self.background_proc.composite_on_background(
                    processed_img, combined_255, raw_bg
                )
                
                # Kontur zeichnen (optional)
                if self.use_contour:
                    preview = self.contour_drawer.process(preview_comp, combined_255)
                else:
                    preview = preview_comp
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
                pass
            
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


# ==================== UI CLASS ====================

class UI(Widget):
    """UI Widget für die Benutzeroberfläche"""
    
    preview_image = ObjectProperty(None)

    def __init__(self, buttonState):
        super(UI, self).__init__()
        self.buttonState = buttonState
        self.stream_proc = None
        self.preprocessing_dropdown = None

    def _create_gradient(self):
        """Erstelle Gradient-Textur für Hintergrund"""
        texture = Texture.create(size=(1, 256), colorfmt='rgba')
        
        buf = []
        for i in range(256):
            ratio = i / 255.0
            r1, g1, b1 = 0.0, 0.0, 0.3
            r2, g2, b2 = 0.02, 0.0, 0.05
            
            r = r1 + (r2 - r1) * ratio
            g = g1 + (g2 - g1) * ratio
            b = b1 + (b2 - b1) * ratio
            
            buf.extend([int(r * 255), int(g * 255), int(b * 255), 255])
        
        buf = bytes(buf)
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture

    # ------------------- Preprocessing Dropdown -------------------
    
    def _create_dropdown_logo_texture(self):
        """Erstelle ein schönes Logo für das Dropdown-Menü"""
        size = 40
        texture = Texture.create(size=(size, size), colorfmt='rgba')
        
        # Erstelle numpy array für das Logo
        logo = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        
        # Zeichne Zahnrad-ähnliches Logo
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            # Äußere Zähne
            x1 = int(center + np.cos(rad) * (size * 0.35))
            y1 = int(center + np.sin(rad) * (size * 0.35))
            x2 = int(center + np.cos(rad) * (size * 0.45))
            y2 = int(center + np.sin(rad) * (size * 0.45))
            
            # Zeichne Linie (vereinfacht als Rechteck)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if 0 <= x1+i < size and 0 <= y1+j < size:
                        logo[y1+j, x1+i] = [50, 150, 255, 255]
                    if 0 <= x2+i < size and 0 <= y2+j < size:
                        logo[y2+j, x2+i] = [50, 150, 255, 255]
        
        # Zeichne Kreis in der Mitte
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist < size * 0.25 and dist > size * 0.15:
                    logo[y, x] = [100, 200, 255, 255]
                elif dist < size * 0.15:
                    logo[y, x] = [30, 100, 200, 255]
        
        # Konvertiere zu Bytes und lade in Textur
        buf = logo.tobytes()
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture
    
    def toggle_preprocessing_dropdown(self, button):
        """Zeige/Verstecke Preprocessing Dropdown Menu"""
        if self.preprocessing_dropdown is None:
            # Berechne dynamische Größe basierend auf Window
            dropdown_width = min(Window.width * 0.25, 350)
            
            # Erstelle Popup statt DropDown für bessere Z-Order Kontrolle
            from kivy.uix.popup import Popup
            
            # Container für alle Toggle-Buttons
            container = BoxLayout(
                orientation='vertical',
                size_hint=(1, 1),
                padding=15,
                spacing=10
            )
            
            # Berechne dynamische Höhe
            num_items = 8  # Header + 7 Features
            item_height = max(40, Window.height * 0.05)
            popup_height = min(num_items * item_height + (num_items - 1) * 10 + 30, Window.height * 0.8)
            
            # Header mit Logo
            header_box = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=item_height,
                spacing=10
            )
            
            # Logo Widget
            logo_widget = KivyImage(
                size_hint=(None, None),
                width=item_height,
                height=item_height
            )
            logo_widget.texture = self._create_dropdown_logo_texture()
            header_box.add_widget(logo_widget)
            
            # Header Text
            header_label = Label(
                text='Preprocessing Controls',
                size_hint=(1, 1),
                font_size=max(14, Window.height * 0.018),
                color=(0.3, 0.8, 1, 1),
                bold=True,
                halign='left',
                valign='middle'
            )
            header_label.bind(size=header_label.setter('text_size'))
            header_box.add_widget(header_label)
            
            # Style für Header Box
            with header_box.canvas.before:
                Color(0.1, 0.15, 0.3, 1)
                header_box.bg_rect = RoundedRectangle(
                    pos=header_box.pos,
                    size=header_box.size,
                    radius=[10,]
                )
            header_box.bind(pos=lambda *args: setattr(header_box.bg_rect, 'pos', header_box.pos))
            header_box.bind(size=lambda *args: setattr(header_box.bg_rect, 'size', header_box.size))
            
            container.add_widget(header_box)
            
            # Feature Toggle Buttons mit Icons
            features = [
                ('Segmentation', 'segmentation'),
                ('Hand Detection', 'hands'),
                ('Pose Detection', 'pose'),
                ('CLAHE Enhancement', 'clahe'),
                ('Brightness Adjust', 'brightness'),
                ('Smart Crop', 'crop'),
                ('Contour Drawing', 'contour'),
            ]
            
            for feature_name, feature_id in features:
                toggle = ToggleButton(
                    text=feature_name,
                    state='down',
                    size_hint_y=None,
                    height=item_height,
                    background_normal='',
                    background_down='',
                    font_size=max(12, Window.height * 0.016),
                    bold=True
                )
                toggle.feature_id = feature_id
                toggle.bind(state=self.on_feature_toggle)
                
                # Dynamische Farben für Toggle
                def update_toggle_color(instance, *args):
                    if instance.state == 'down':
                        instance.background_color = (0.1, 0.6, 0.3, 1)
                    else:
                        instance.background_color = (0.3, 0.3, 0.4, 1)
                
                toggle.bind(state=update_toggle_color)
                update_toggle_color(toggle)
                
                container.add_widget(toggle)
            
            # Style für Container
            with container.canvas.before:
                Color(0.08, 0.12, 0.22, 0.98)
                container.bg_rect = RoundedRectangle(
                    pos=container.pos,
                    size=container.size,
                    radius=[15,]
                )
                Color(0.25, 0.4, 0.8, 0.6)
                container.border = RoundedRectangle(
                    pos=(container.x + 1, container.y + 1),
                    size=(container.width - 2, container.height - 2),
                    radius=[14,]
                )
            
            def update_container_graphics(instance, *args):
                container.bg_rect.pos = instance.pos
                container.bg_rect.size = instance.size
                container.border.pos = (instance.x + 1, instance.y + 1)
                container.border.size = (instance.width - 2, instance.height - 2)
            
            container.bind(pos=update_container_graphics)
            container.bind(size=update_container_graphics)
            
            # Erstelle Popup mit Container
            self.preprocessing_dropdown = Popup(
                title='',
                content=container,
                size_hint=(None, None),
                size=(dropdown_width, popup_height),
                separator_height=0,
                background='',
                background_color=(0, 0, 0, 0),
                auto_dismiss=True,
                pos_hint={'center_x': 0.5, 'top': 0.95}
            )
        
        # Toggle Popup
        if self.preprocessing_dropdown._window:
            self.preprocessing_dropdown.dismiss()
        else:
            self.preprocessing_dropdown.open()
    
    def _on_dropdown_dismiss(self):
        """Callback wenn Dropdown geschlossen wird"""
        pass
    
    def on_feature_toggle(self, instance, value):
        """Callback für Feature Toggle Buttons"""
        if not self.stream_proc:
            return
        
        is_enabled = (value == 'down')
        feature_id = instance.feature_id
        
        # Toggle entsprechendes Feature im StreamProcessor
        if feature_id == 'segmentation':
            self.stream_proc.toggle_segmentation(is_enabled)
        elif feature_id == 'hands':
            self.stream_proc.toggle_hands(is_enabled)
        elif feature_id == 'pose':
            self.stream_proc.toggle_pose(is_enabled)
        elif feature_id == 'clahe':
            self.stream_proc.toggle_clahe(is_enabled)
        elif feature_id == 'brightness':
            self.stream_proc.toggle_brightness(is_enabled)
        elif feature_id == 'crop':
            self.stream_proc.toggle_crop(is_enabled)
        elif feature_id == 'contour':
            self.stream_proc.toggle_contour(is_enabled)

    # ------------------- Gesture Display Updates -------------------
    
    def update_gesture_guesses(self, guesses):
        try:
            if len(guesses) > 0:
                self.gesture_label_1.text = guesses[0][0]
                self.gesture_confidence_1.text = f"{guesses[0][1]:.1f}%"
            else:
                self.gesture_label_1.text = "---"
                self.gesture_confidence_1.text = "0%"
            
            if len(guesses) > 1:
                self.gesture_label_2.text = guesses[1][0]
                self.gesture_confidence_2.text = f"{guesses[1][1]:.1f}%"
            else:
                self.gesture_label_2.text = "---"
                self.gesture_confidence_2.text = "0%"
            
            if len(guesses) > 2:
                self.gesture_label_3.text = guesses[2][0]
                self.gesture_confidence_3.text = f"{guesses[2][1]:.1f}%"
            else:
                self.gesture_label_3.text = "---"
                self.gesture_confidence_3.text = "0%"
        except Exception as e:
            consoleInterface.print_error(f"Error updating gesture guesses: {e}")

    def clear_gesture_guesses(self):
        self.gesture_label_1.text = "---"
        self.gesture_confidence_1.text = "0%"
        self.gesture_label_2.text = "---"
        self.gesture_confidence_2.text = "0%"
        self.gesture_label_3.text = "---"
        self.gesture_confidence_3.text = "0%"
    
    # ------------------- Toggle Button Callbacks -------------------
    
    def toggle_postprocessing(self, instance):
        if instance.state == 'down':
            self.buttonState.usePostProcessing = True
            consoleInterface.print_status("Postprocessing activated")
        else:
            self.buttonState.usePostProcessing = False
            consoleInterface.print_status("Postprocessing deactivated")
    
    def toggle_TextToSpeech(self, instance):
        if instance.state == 'down':
            self.buttonState.useTextToSpeech = True
            consoleInterface.print_status("TextToSpeech activated")
        else:
            self.buttonState.useTextToSpeech = False
            consoleInterface.print_status("TextToSpeech deactivated")

    # ------------------- Stream Processor Verbindung -------------------
    
    def set_stream_processor(self, sp: StreamProcessor):
        self.stream_proc = sp

    # ------------------- Preview Update -------------------
    
    def update_preview_texture(self, dt):
        if self.stream_proc is None:
            return
        
        preview = self.stream_proc.get_preview()
        if preview is None:
            return
        
        try:
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            
            tex = self.preview_image.texture
            if tex is None or tex.width != w or tex.height != h:
                tex = Texture.create(size=(w, h), colorfmt='rgb')
                tex.flip_vertical()
                self.preview_image.texture = tex
            
            buf = rgb.tobytes()
            self.preview_image.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.preview_image.canvas.ask_update()
        except Exception as e:
            consoleInterface.print_error("Preview update error:", e)


# ==================== MAIN APP ====================

class MyApp(App):
    """Haupt-Kivy-App"""
    
    fps = NumericProperty(25.0)

    def __init__(self, state):
        super(MyApp, self).__init__()
        self.buttonState = state

    def build(self):
        Window.clearcolor = (0, 0, 75/255.0, 0.5)
        self.root = UI(self.buttonState)
        return self.root

    def on_start(self):
        # Erstelle StreamProcessor
        self.sp = StreamProcessor(
            self.buttonState,
            ai_out_dir="preprocessing/out",
            ai_w=210,
            ai_h=300,
            target_fps=self.fps
        )
        
        # Verbinde mit UI
        self.root.set_stream_processor(self.sp)
        
        # Starte Stream
        self.sp.start(show_preview=False)
        
        # Schedule UI Updates
        Clock.schedule_interval(self.root.update_preview_texture, 1.0 / 30.0)
        Clock.schedule_interval(self.while_running, 0.1)
    
    def while_running(self, dt):
        if not hasattr(self, 'root') or not hasattr(self, 'buttonState'):
            return
        
        if self.buttonState.gotGeasture:
            self.root.clear_gesture_guesses()
            self.root.update_gesture_guesses(self.buttonState.guesses)
            self.buttonState.gotGeasture = False

    def on_stop(self):
        try:
            if hasattr(self, 'sp') and self.sp is not None:
                self.sp.stop()
        except Exception as e:
            consoleInterface.print_error("Error stopping stream processor:", e)


# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    class State():
        def __init__(self):
            self.isRunning = True
            self.usePreProcessing = True
            self.usePostProcessing = True
            self.useTextToSpeech = True
            self.gotGeasture = False
            self.guesses = [["", 0.0], ["", 0.0], ["", 0.0]]

    state = State()
    app = MyApp(state)
    app.run()