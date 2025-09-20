from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.core.window import Window

Builder.load_file('uiDesign.kv')
class UI(Widget): 
    pass


class MyApp(App):
    def build(self):
        Window.clearcolor = (0, 0, 75/255.0, 0.5) # set the background color for the whole app (every other color from .kv file can override this)
        return UI()


if __name__ == '__main__':
    MyApp().run()
