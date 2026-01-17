"""
processor_manager.py
Manages all image processing processors and their initialization
"""


class ProcessorManager:
    """Manages initialization and access to all image processors"""
    
    def __init__(self):
        # Import processors
        from preprocessing.brightness_processor import BrightnessProcessor
        from preprocessing.clahe_processor import CLAHEProcessor
        from preprocessing.crop_resize_processor import CropResizeProcessor
        from preprocessing.mask_combiner_processor import MaskCombinerProcessor
        from preprocessing.background_removal_processor import BackgroundRemovalProcessor
        from preprocessing.contour_drawer_processor import ContourDrawerProcessor
        from preprocessing.segmentation_processor import SegmentationProcessor
        from preprocessing.hand_detection_processor import HandsDetectionProcessor
        from preprocessing.face_detection_processor import FaceDetectionProcessor
        from preprocessing.pose_detection_processor import PoseDetectionProcessor
        from .. import consoleInterface
        self.console = consoleInterface
            
        
        # Initialize all processors
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
        self.hand_proc = HandsDetectionProcessor(confidence=0.5)
        self.face_proc = FaceDetectionProcessor(confidence=0.5)
        self.pose_proc = PoseDetectionProcessor(confidence=0.5)
    
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
    
    def get_face_processor(self):
        return self.face_proc
    
    def get_pose_processor(self):
        return self.pose_proc
