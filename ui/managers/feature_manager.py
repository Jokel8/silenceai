"""
feature_manager.py
Manages feature toggles and their state
"""


class FeatureManager:
    """Manages feature toggle states"""
    
    def __init__(self, console):
        self.console = console
        
        # Feature toggles
        self.use_segmentation = True
        self.use_hands = True
        self.use_pose = True
        self.use_clahe = True
        self.use_brightness = True
        self.use_crop = True
        self.use_contour = True
    
    def toggle_segmentation(self, enabled):
        self.use_segmentation = enabled
        self.console.print_status(f"Segmentation: {'ON' if enabled else 'OFF'}")
    
    def toggle_hands(self, enabled):
        self.use_hands = enabled
        self.console.print_status(f"Hand Detection: {'ON' if enabled else 'OFF'}")
    
    def toggle_pose(self, enabled):
        self.use_pose = enabled
        self.console.print_status(f"Pose Detection: {'ON' if enabled else 'OFF'}")
    
    def toggle_clahe(self, enabled):
        self.use_clahe = enabled
        self.console.print_status(f"CLAHE: {'ON' if enabled else 'OFF'}")
    
    def toggle_brightness(self, enabled):
        self.use_brightness = enabled
        self.console.print_status(f"Brightness Adjustment: {'ON' if enabled else 'OFF'}")
    
    def toggle_crop(self, enabled):
        self.use_crop = enabled
        self.console.print_status(f"Smart Crop: {'ON' if enabled else 'OFF'}")
    
    def toggle_contour(self, enabled):
        self.use_contour = enabled
        self.console.print_status(f"Contour Drawing: {'ON' if enabled else 'OFF'}")
