from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import ObjectProperty, NumericProperty
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import os
# import your StreamProcessor module
try:
    from userInterfaces.stream_processor import StreamProcessor
    Builder.load_file('userInterfaces/uiDesign.kv')
    from userInterfaces import consoleInterface

except ImportError:
    from stream_processor import StreamProcessor
    import consoleInterface
    Builder.load_file('uiDesign.kv')




class UI(Widget):

    # preview image widget (Image in kv)
    preview_image = ObjectProperty(None)

    def __init__(self, buttonState):
        super(UI, self).__init__()
        self.buttonState = buttonState

    # runtime object: StreamProcessor instance (set by App)
    stream_proc = None

    def _create_gradient(self):
        #"""Create a gradient texture for the background"""
        
        # Create texture
        texture = Texture.create(size=(1, 256), colorfmt='rgba')
        
        # Create gradient data - from blue to dark purple/black
        buf = []
        for i in range(256):
            # Gradient from blue (top) to dark (bottom)
            ratio = i / 255.0
            
            # Start color (blue) - kannst du anpassen
            r1, g1, b1 = 0.0, 0.0, 0.3  # Dunkles Blau
            
            # End color (very dark purple/black)
            r2, g2, b2 = 0.02, 0.0, 0.05  # Fast schwarz mit leichtem Lila-Stich
            
            # Interpolate
            r = r1 + (r2 - r1) * ratio
            g = g1 + (g2 - g1) * ratio
            b = b1 + (b2 - b1) * ratio
            
            # Convert to bytes
            buf.extend([int(r * 255), int(g * 255), int(b * 255), 255])
        
        # Apply to texture
        buf = bytes(buf)
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        
        return texture

    def update_gesture_predictions(self, predictions):
        """Update the top 3 gesture predictions in the UI
        
        Args:
            predictions: List of tuples [(gesture_name, confidence), ...]
                        sorted by confidence in descending order
        """
        try:
            # Update first prediction
            if len(predictions) > 0:
                self.gesture_label_1.text = predictions[0][0]
                self.gesture_confidence_1.text = f"{predictions[0][1]:.1f}%"
            else:
                self.gesture_label_1.text = "---"
                self.gesture_confidence_1.text = "0%"
            
            # Update second prediction
            if len(predictions) > 1:
                self.gesture_label_2.text = predictions[1][0]
                self.gesture_confidence_2.text = f"{predictions[1][1]:.1f}%"
            else:
                self.gesture_label_2.text = "---"
                self.gesture_confidence_2.text = "0%"
            
            # Update third prediction
            if len(predictions) > 2:
                self.gesture_label_3.text = predictions[2][0]
                self.gesture_confidence_3.text = f"{predictions[2][1]:.1f}%"
            else:
                self.gesture_label_3.text = "---"
                self.gesture_confidence_3.text = "0%"
                
        except Exception as e:
            consoleInterface.print_error(f"Error updating gesture predictions: {e}")

    def clear_gesture_predictions(self):
        """Clear all gesture prediction displays"""
        self.gesture_label_1.text = "---"
        self.gesture_confidence_1.text = "0%"
        self.gesture_label_2.text = "---"
        self.gesture_confidence_2.text = "0%"
        self.gesture_label_3.text = "---"
        self.gesture_confidence_3.text = "0%"
    
    

    def toggle_preprocessing(self, instance):
        # instance is the ToggleButton — check instance.state ('down' or 'normal')
        if instance.state == 'down':
            self.buttonState.usePreProcessing = True
            consoleInterface.print_status("Preprocessing activated")
        else:
            self.buttonState.usePreProcessing = False
            consoleInterface.print_status("Preprocessing deactivated")
    def toggle_postprocessing(self, instance):
        # instance is the ToggleButton — check instance.state ('down' or 'normal')
        if instance.state == 'down':
            self.buttonState.usePostProcessing = True
            consoleInterface.print_status("Postprocessing activated")
        else:
            self.buttonState.usePostProcessing = False
            consoleInterface.print_status("Postprocessing deactivated")
    def toggle_TextToSpeech(self, instance):
        # instance is the ToggleButton — check instance.state ('down' or 'normal')
        if instance.state == 'down':
            self.buttonState.useTextToSpeech = True
            consoleInterface.print_status("TextToSpeech activated")
        else:
            self.buttonState.useTextToSpeech = False
            consoleInterface.print_status("TextToSpeech deactivated")

    def set_stream_processor(self, sp: StreamProcessor):
        """Called by App after creating StreamProcessor so UI can access it."""
        self.stream_proc = sp

    def update_preview_texture(self, dt):
        """Called periodically by Clock to fetch the latest preview frame and update the Kivy Image texture."""
        if self.stream_proc is None:
            return
        preview = self.stream_proc.get_preview()  # expected BGR numpy array or None
        if preview is None:
            return
        # preview is BGR (H,W,3) uint8, convert to RGB bytes expected by Kivy
        try:
            import cv2
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # create texture if not exists or if size changed
            tex = self.preview_image.texture
            if tex is None or tex.width != w or tex.height != h:
                tex = Texture.create(size=(w, h), colorfmt='rgb')
                tex.flip_vertical()  # OpenCV origin is top-left; Kivy textures often expect bottom-left
                self.preview_image.texture = tex
            # blit the buffer (ensure contiguous)
            buf = rgb.tobytes()
            # update texture (note colorfmt must match)
            self.preview_image.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.preview_image.canvas.ask_update()
        except Exception as e:
            # don't crash UI for one conversion error
            consoleInterface.print_error("Preview update error:", e)


class MyApp(App):
    fps = NumericProperty(25.0)

    def __init__(self, state):
        super(MyApp, self).__init__()
        self.buttonState = state

    def build(self):
        Window.clearcolor = (0, 0, 75/255.0, 0.5)
        self.root = UI(self.buttonState)
        return self.root

    def on_start(self):
        # create and start StreamProcessor
        # make sure stream_processor.py is in same folder
        self.sp = StreamProcessor(self.buttonState, ai_out_dir="preprocessing/out", ai_w=210, ai_h=300, target_fps=self.fps)
        # attach it to UI so UI can get preview frames
        self.root.set_stream_processor(self.sp)
        # start the stream (do not show OpenCV windows since Kivy will display preview)
        self.sp.start(show_preview=False)
        # schedule the UI preview update at ~30Hz for responsive UI (can be lower)
        Clock.schedule_interval(self.root.update_preview_texture, 1.0 / 30.0)

    def on_stop(self):
        # stop stream gracefully
        try:
            if hasattr(self, 'sp') and self.sp is not None:
                self.sp.stop()
        except Exception as e:
            consoleInterface.print_error("Error stopping stream processor:", e)

if __name__ == '__main__':
    class State():
        def __init__(self):
            self.isRunning = True
            self.usePreProcessing = True
            self.usePostProcessing = True
            self.useTextToSpeech = True
            self.gotGeasture = False

    state = State()
    MyApp(state).run()










# from kivy.app import App
# from kivy.uix.widget import Widget
# from kivy.lang import Builder
# from kivy.core.window import Window
# from kivy.properties import ObjectProperty

# Builder.load_file('uiDesign.kv')
# class UI(Widget): 
#     # Define properties for accessing widgets in .kv file
#     toggle_preprocessing_btn = ObjectProperty(None)
#     toggle_postprocessing_btn = ObjectProperty(None)
#     toggle_videototext_btn = ObjectProperty(None)
#     toggle_texttospeech_btn = ObjectProperty(None)
#     # def for toggle buttons and other stuff
#     def toggle_preprocessing(self, toggle_preprocessing_btn):
#         if toggle_preprocessing_btn == 'down':
#             print("Preprocessing is enabled")
#         else:
#             print("Preprocessing is disabled")
        
    

# class MyApp(App):
#     def build(self):
#         Window.clearcolor = (0, 0, 75/255.0, 0.5) # set the background color for the whole app (every other color from .kv file can override this)
#         return UI()


# if __name__ == '__main__':
#     MyApp().run()
