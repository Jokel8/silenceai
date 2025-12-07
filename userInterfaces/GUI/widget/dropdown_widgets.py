"""
dropdown_widgets.py
Custom dropdown widgets with proper styling
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle
from kivy.core.window import Window
from kivy.graphics.texture import Texture
import numpy as np


class StyledToggleButton(ToggleButton):
    """Toggle button with 3D styling"""
    
    def __init__(self, **kwargs):
        super(StyledToggleButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_down = ''
        self.background_color = [0, 0, 0, 0]  # Transparent
        self.color = [1, 1, 1, 1]
        self.bold = True
        
        # Initialize graphics immediately
        self._setup_graphics()
        
        # Bind events
        self.bind(state=self._update_graphics)
        self.bind(pos=self._update_graphics)
        self.bind(size=self._update_graphics)
    
    def _setup_graphics(self):
        """Initial graphics setup"""
        with self.canvas.before:
            self.shadow_color = Color(0, 0, 0, 0.3)
            self.shadow_rect = RoundedRectangle(
                pos=(self.x + 3, self.y - 3),
                size=self.size,
                radius=[15,]
            )
            
            self.bg_color = Color(0.2, 0.3, 0.8, 1)
            self.bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[15,]
            )
            
            self.highlight_color = Color(1, 1, 1, 0.1)
            self.highlight_rect = RoundedRectangle(
                pos=(self.x, self.y + self.height * 0.7),
                size=(self.width, self.height * 0.3),
                radius=[15, 15, 0, 0]
            )
            
            self.border_color = Color(0.4, 0.5, 1, 0.3)
            self.border_line = Line(
                width=2,
                rounded_rectangle=(self.x + 2, self.y + 2, self.width - 4, self.height - 4, 13)
            )
    
    def _update_graphics(self, *args):
        """Update graphics on state/pos/size change"""
        # Update shadow
        self.shadow_color.rgba = [0, 0, 0, 0.5 if self.state == 'down' else 0.3]
        self.shadow_rect.pos = (self.x + 3, self.y - 3)
        self.shadow_rect.size = self.size
        
        # Update background
        if self.state == 'down':
            self.bg_color.rgba = [0.1, 0.8, 0.4, 1]
        else:
            self.bg_color.rgba = [0.2, 0.3, 0.8, 1]
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        
        # Update highlight
        self.highlight_rect.pos = (self.x, self.y + self.height * 0.7)
        self.highlight_rect.size = (self.width, self.height * 0.3)
        
        # Update border
        if self.state == 'down':
            self.border_color.rgba = [0.2, 1, 0.5, 0.5]
        else:
            self.border_color.rgba = [0.4, 0.5, 1, 0.3]
        self.border_line.rounded_rectangle = (self.x + 2, self.y + 2, self.width - 4, self.height - 4, 13)


class StyledSelectionButton(Button):
    """Selection button with 3D styling"""
    
    def __init__(self, **kwargs):
        super(StyledSelectionButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_down = ''
        self.background_color = [0, 0, 0, 0]  # Transparent
        self.color = [1, 1, 1, 1]
        self.bold = True
        self.is_selected = False
        
        # Initialize graphics immediately
        self._setup_graphics()
        
        # Bind events
        self.bind(pos=self._update_graphics)
        self.bind(size=self._update_graphics)
    
    def _setup_graphics(self):
        """Initial graphics setup"""
        with self.canvas.before:
            self.shadow_color = Color(0, 0, 0, 0.3)
            self.shadow_rect = RoundedRectangle(
                pos=(self.x + 3, self.y - 3),
                size=self.size,
                radius=[15,]
            )
            
            self.bg_color = Color(0.2, 0.3, 0.8, 1)
            self.bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[15,]
            )
            
            self.highlight_color = Color(1, 1, 1, 0.1)
            self.highlight_rect = RoundedRectangle(
                pos=(self.x, self.y + self.height * 0.7),
                size=(self.width, self.height * 0.3),
                radius=[15, 15, 0, 0]
            )
            
            self.border_color = Color(0.4, 0.5, 1, 0.3)
            self.border_line = Line(
                width=2,
                rounded_rectangle=(self.x + 2, self.y + 2, self.width - 4, self.height - 4, 13)
            )
    
    def set_selected(self, selected):
        self.is_selected = selected
        self._update_graphics()
    
    def _update_graphics(self, *args):
        """Update graphics based on selection state"""
        # Update shadow
        self.shadow_rect.pos = (self.x + 3, self.y - 3)
        self.shadow_rect.size = self.size
        
        # Update background
        if self.is_selected:
            self.bg_color.rgba = [0.1, 0.8, 0.4, 1]
        else:
            self.bg_color.rgba = [0.2, 0.3, 0.8, 1]
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        
        # Update highlight
        self.highlight_rect.pos = (self.x, self.y + self.height * 0.7)
        self.highlight_rect.size = (self.width, self.height * 0.3)
        
        # Update border
        if self.is_selected:
            self.border_color.rgba = [0.2, 1, 0.5, 0.5]
        else:
            self.border_color.rgba = [0.4, 0.5, 1, 0.3]
        self.border_line.rounded_rectangle = (self.x + 2, self.y + 2, self.width - 4, self.height - 4, 13)


class StyledContainer(BoxLayout):
    """Styled container for dropdown content"""
    
    def __init__(self, **kwargs):
        super(StyledContainer, self).__init__(**kwargs)
        
        # Setup canvas
        with self.canvas.before:
            self.bg_color = Color(0.08, 0.12, 0.22, 0.98)
            self.bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[15,]
            )
            self.border_color = Color(0.25, 0.4, 0.8, 0.6)
            self.border_line = Line(
                width=2,
                rounded_rectangle=(self.x + 1, self.y + 1, self.width - 2, self.height - 2, 14)
            )
        
        self.bind(pos=self._update_graphics)
        self.bind(size=self._update_graphics)
    
    def _update_graphics(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        self.border_line.rounded_rectangle = (self.x + 1, self.y + 1, self.width - 2, self.height - 2, 14)


class StyledHeader(BoxLayout):
    """Styled header box"""
    
    def __init__(self, **kwargs):
        super(StyledHeader, self).__init__(**kwargs)
        
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.15, 0.3, 1)
            self.bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[10,]
            )
        
        self.bind(pos=self._update_graphics)
        self.bind(size=self._update_graphics)
    
    def _update_graphics(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


class PreprocessingDropdown:
    """Creates preprocessing settings dropdown"""
    
    @staticmethod
    def create(callback):
        dropdown_width = min(Window.width * 0.25, 350)
        item_height = max(40, Window.height * 0.05)
        num_items = 8  # Header + 7 features
        popup_height = min(num_items * item_height + (num_items - 1) * 10 + 30, Window.height * 0.8)
        
        container = StyledContainer(
            orientation='vertical',
            size_hint=(1, 1),
            padding=15,
            spacing=10
        )
        
        # Header
        header = StyledHeader(
            orientation='horizontal',
            size_hint_y=None,
            height=item_height,
            spacing=10
        )
        
        # Logo
        logo_widget = KivyImage(
            size_hint=(None, None),
            width=item_height,
            height=item_height
        )
        logo_widget.texture = PreprocessingDropdown._create_gear_logo()
        header.add_widget(logo_widget)
        
        # Title
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
        header.add_widget(header_label)
        
        container.add_widget(header)
        
        # Features
        features = [
            ('Segmentation', 'segmentation', True),
            ('Hand Detection', 'hands', True),
            ('Pose Detection', 'pose', True),
            ('CLAHE Enhancement', 'clahe', True),
            ('Brightness Adjust', 'brightness', True),
            ('Smart Crop', 'crop', True),
            ('Contour Drawing', 'contour', True),
        ]
        
        for display_name, feature_id, initial_state in features:
            toggle = StyledToggleButton(
                text=display_name,
                state='down' if initial_state else 'normal',
                size_hint_y=None,
                height=item_height,
                font_size=max(12, Window.height * 0.016)
            )
            toggle.feature_id = feature_id
            toggle.bind(state=callback)
            container.add_widget(toggle)
        
        popup = Popup(
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
        
        return popup
    
    @staticmethod
    def _create_gear_logo():
        size = 40
        texture = Texture.create(size=(size, size), colorfmt='rgba')
        logo = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        
        # Draw gear teeth
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            x1 = int(center + np.cos(rad) * (size * 0.35))
            y1 = int(center + np.sin(rad) * (size * 0.35))
            x2 = int(center + np.cos(rad) * (size * 0.45))
            y2 = int(center + np.sin(rad) * (size * 0.45))
            
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if 0 <= x1+i < size and 0 <= y1+j < size:
                        logo[y1+j, x1+i] = [50, 150, 255, 255]
                    if 0 <= x2+i < size and 0 <= y2+j < size:
                        logo[y2+j, x2+i] = [50, 150, 255, 255]
        
        # Draw center circle
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist < size * 0.25 and dist > size * 0.15:
                    logo[y, x] = [100, 200, 255, 255]
                elif dist < size * 0.15:
                    logo[y, x] = [30, 100, 200, 255]
        
        buf = logo.tobytes()
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture


class ModelSelectionDropdown:
    """Creates model selection dropdown"""
    
    @staticmethod
    def create(callback):
        dropdown_width = min(Window.width * 0.25, 350)
        item_height = max(40, Window.height * 0.05)
        num_items = 5  # Header + 4 models
        popup_height = min(num_items * item_height + (num_items - 1) * 10 + 30, Window.height * 0.8)
        
        container = StyledContainer(
            orientation='vertical',
            size_hint=(1, 1),
            padding=15,
            spacing=10
        )
        
        # Header
        header = StyledHeader(
            orientation='horizontal',
            size_hint_y=None,
            height=item_height,
            spacing=10
        )
        
        # Logo
        logo_widget = KivyImage(
            size_hint=(None, None),
            width=item_height,
            height=item_height
        )
        logo_widget.texture = ModelSelectionDropdown._create_network_logo()
        header.add_widget(logo_widget)
        
        # Title
        header_label = Label(
            text='Model Selection',
            size_hint=(1, 1),
            font_size=max(14, Window.height * 0.018),
            color=(0.3, 0.8, 1, 1),
            bold=True,
            halign='left',
            valign='middle'
        )
        header_label.bind(size=header_label.setter('text_size'))
        header.add_widget(header_label)
        
        container.add_widget(header)
        
        # Models
        models = [
            ('Model v1.0', 'v1.0', True),
            ('Model v1.1', 'v1.1', False),
            ('Model v2.0', 'v2.0', False),
            ('Model v2.1 Beta', 'v2.1', False),
        ]
        
        buttons = []
        for display_name, model_id, is_selected in models:
            btn = StyledSelectionButton(
                text=display_name,
                size_hint_y=None,
                height=item_height,
                font_size=max(12, Window.height * 0.016)
            )
            btn.model_id = model_id
            btn.set_selected(is_selected)
            
            def make_callback(button, all_buttons):
                def on_click(instance):
                    for b in all_buttons:
                        b.set_selected(False)
                    button.set_selected(True)
                    callback(button)
                return on_click
            
            btn.bind(on_release=make_callback(btn, buttons))
            buttons.append(btn)
            container.add_widget(btn)
        
        popup = Popup(
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
        
        return popup
    
    @staticmethod
    def _create_network_logo():
        size = 40
        texture = Texture.create(size=(size, size), colorfmt='rgba')
        logo = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        
        # Neural network nodes
        nodes = [
            (center - 10, center - 10),
            (center + 10, center - 10),
            (center, center),
            (center - 10, center + 10),
            (center + 10, center + 10)
        ]
        
        # Draw connections
        for i, (x1, y1) in enumerate(nodes):
            for j, (x2, y2) in enumerate(nodes):
                if i < j:
                    steps = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                    for step in range(steps):
                        t = step / steps
                        x = int(x1 + (x2-x1) * t)
                        y = int(y1 + (y2-y1) * t)
                        if 0 <= x < size and 0 <= y < size:
                            logo[y, x] = [80, 180, 255, 180]
        
        # Draw nodes
        for x, y in nodes:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        if 0 <= x+dx < size and 0 <= y+dy < size:
                            logo[y+dy, x+dx] = [120, 220, 255, 255]
        
        buf = logo.tobytes()
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture


class PostprocessingDropdown:
    """Creates postprocessing settings dropdown"""
    
    @staticmethod
    def create(callback):
        dropdown_width = min(Window.width * 0.25, 350)
        item_height = max(40, Window.height * 0.05)
        num_items = 3  # Header + 2 features
        popup_height = min(num_items * item_height + (num_items - 1) * 10 + 30, Window.height * 0.8)
        
        container = StyledContainer(
            orientation='vertical',
            size_hint=(1, 1),
            padding=15,
            spacing=10
        )
        
        # Header
        header = StyledHeader(
            orientation='horizontal',
            size_hint_y=None,
            height=item_height,
            spacing=10
        )
        
        # Logo
        logo_widget = KivyImage(
            size_hint=(None, None),
            width=item_height,
            height=item_height
        )
        logo_widget.texture = PostprocessingDropdown._create_text_logo()
        header.add_widget(logo_widget)
        
        # Title
        header_label = Label(
            text='Postprocessing Settings',
            size_hint=(1, 1),
            font_size=max(14, Window.height * 0.018),
            color=(0.3, 0.8, 1, 1),
            bold=True,
            halign='left',
            valign='middle'
        )
        header_label.bind(size=header_label.setter('text_size'))
        header.add_widget(header_label)
        
        container.add_widget(header)
        
        # Features
        features = [
            ('Grammar Correction', 'grammar', True),
            ('Text to Speech', 'tts', True),
        ]
        
        for display_name, feature_id, initial_state in features:
            toggle = StyledToggleButton(
                text=display_name,
                state='down' if initial_state else 'normal',
                size_hint_y=None,
                height=item_height,
                font_size=max(12, Window.height * 0.016)
            )
            toggle.feature_id = feature_id
            toggle.bind(state=callback)
            container.add_widget(toggle)
        
        popup = Popup(
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
        
        return popup
    
    @staticmethod
    def _create_text_logo():
        size = 40
        texture = Texture.create(size=(size, size), colorfmt='rgba')
        logo = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        
        # Draw text-like bars
        bar_positions = [
            (center - 12, center - 8, 8, 3),
            (center - 12, center - 2, 12, 3),
            (center - 12, center + 4, 10, 3),
            (center + 4, center - 8, 8, 3),
            (center + 4, center - 2, 10, 3),
            (center + 4, center + 4, 6, 3)
        ]
        
        for x, y, w, h in bar_positions:
            for dy in range(h):
                for dx in range(w):
                    if 0 <= x+dx < size and 0 <= y+dy < size:
                        logo[y+dy, x+dx] = [100, 200, 120, 255]
        
        buf = logo.tobytes()
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture