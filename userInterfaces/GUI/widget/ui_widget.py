"""
ui_widget.py
Main UI Widget class handling user interface and interactions
"""

from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
import cv2

try:
    from userInterfaces import consoleInterface
    from userInterfaces.GUI.widget.dropdown_widgets import (
        PreprocessingDropdown, 
        ModelSelectionDropdown, 
        PostprocessingDropdown
    )
except ImportError:
    import consoleInterface
    from .dropdown_widgets import (
        PreprocessingDropdown, 
        ModelSelectionDropdown, 
        PostprocessingDropdown
    )


class UI(Widget):
    """UI Widget for the user interface"""
    
    preview_image = ObjectProperty(None)
    gesture_label_1 = ObjectProperty(None)
    gesture_confidence_1 = ObjectProperty(None)
    gesture_label_2 = ObjectProperty(None)
    gesture_confidence_2 = ObjectProperty(None)
    gesture_label_3 = ObjectProperty(None)
    gesture_confidence_3 = ObjectProperty(None)

    def __init__(self, buttonState):
        super(UI, self).__init__()
        self.buttonState = buttonState
        self.stream_proc = None
        self.preprocessing_dropdown = None
        self.model_dropdown = None
        self.postprocessing_dropdown = None

    def _create_gradient(self):
        """Create gradient texture for background"""
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
    
    def toggle_preprocessing_dropdown(self, button):
        """Show/Hide Preprocessing Dropdown Menu"""
        if self.preprocessing_dropdown is None:
            self.preprocessing_dropdown = PreprocessingDropdown.create(
                self.on_feature_toggle
            )
        
        if self.preprocessing_dropdown._window:
            self.preprocessing_dropdown.dismiss()
        else:
            self.preprocessing_dropdown.open()
    
    def on_feature_toggle(self, instance, value):
        """Callback for Feature Toggle Buttons"""
        if not self.stream_proc:
            return
        
        is_enabled = (value == 'down')
        feature_id = instance.feature_id
        
        # Toggle corresponding feature in StreamProcessor
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

    # ------------------- Model Selection Dropdown -------------------
    
    def toggle_model_dropdown(self, button):
        """Show/Hide Model Selection Dropdown Menu"""
        if self.model_dropdown is None:
            self.model_dropdown = ModelSelectionDropdown.create(
                self.on_model_select
            )
        
        if self.model_dropdown._window:
            self.model_dropdown.dismiss()
        else:
            self.model_dropdown.open()
    
    def on_model_select(self, instance):
        """Callback for Model Selection"""
        model_id = instance.model_id
        consoleInterface.print_status(f"Selected model: {model_id}")
        
        # Update button state
        if hasattr(self, 'buttonState'):
            self.buttonState.selectedModel = model_id
        
        # Close dropdown after selection
        if self.model_dropdown:
            self.model_dropdown.dismiss()

    # ------------------- Postprocessing Dropdown -------------------
    
    def toggle_postprocessing_dropdown(self, button):
        """Show/Hide Postprocessing Dropdown Menu"""
        if self.postprocessing_dropdown is None:
            self.postprocessing_dropdown = PostprocessingDropdown.create(
                self.on_postprocessing_toggle
            )
        
        if self.postprocessing_dropdown._window:
            self.postprocessing_dropdown.dismiss()
        else:
            self.postprocessing_dropdown.open()
    
    def on_postprocessing_toggle(self, instance, value):
        """Callback for Postprocessing Toggle Buttons"""
        is_enabled = (value == 'down')
        feature_id = instance.feature_id
        
        if feature_id == 'grammar':
            self.buttonState.usePostProcessing = is_enabled
            consoleInterface.print_status(f"Grammar Correction: {'ON' if is_enabled else 'OFF'}")
        elif feature_id == 'tts':
            self.buttonState.useTextToSpeech = is_enabled
            consoleInterface.print_status(f"Text to Speech: {'ON' if is_enabled else 'OFF'}")

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

    # ------------------- Stream Processor Connection -------------------
    
    def set_stream_processor(self, sp):
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
            consoleInterface.print_error(f"Preview update error: {e}")