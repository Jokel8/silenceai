"""
main_app.py
Main Kivy application entry point
"""

from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import NumericProperty
from kivy.clock import Clock

try:
    from userInterfaces import consoleInterface
    from userInterfaces.GUI.widget.ui_widget import UI
    from userInterfaces.GUI.stream_processor import StreamProcessor
    Builder.load_file('userInterfaces/uiDesign.kv')
except ImportError:
    import consoleInterface
    from GUI.widget.ui_widget import UI
    from GUI.stream_processor import StreamProcessor
    Builder.load_file('uiDesign.kv')


class MyApp(App):
    """Main Kivy Application"""
    
    fps = NumericProperty(25.0)

    def __init__(self, state):
        super(MyApp, self).__init__()
        self.buttonState = state

    def build(self):
        Window.clearcolor = (0, 0, 75/255.0, 0.5)
        self.root = UI(self.buttonState)
        return self.root

    def on_start(self):
        # Create StreamProcessor
        self.sp = StreamProcessor(
            self.buttonState,
            ai_out_dir="preprocessing/out",
            ai_w=210,
            ai_h=300,
            target_fps=self.fps
        )
        
        # Connect with UI
        self.root.set_stream_processor(self.sp)
        
        # Start stream
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
            consoleInterface.print_error(f"Error stopping stream processor: {e}")


if __name__ == '__main__':
    class State():
        def __init__(self):
            self.isRunning = True
            self.usePreProcessing = True
            self.usePostProcessing = True
            self.useTextToSpeech = True
            self.gotGeasture = False
            self.guesses = [["", 0.0], ["", 0.0], ["", 0.0]]
            self.selectedModel = "v1.0"

    state = State()
    app = MyApp(state)
    app.run()