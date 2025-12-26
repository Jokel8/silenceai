"""
core_processing.py
Core AI model processing and analysis logic
"""

import numpy as np
import tensorflow as tf
import joblib
import os


class GestureAnalyzer:
    """Analyzes hand keypoints using trained AI model"""
    
    def __init__(self, model_path="training/models/gesture_model_phoenix2.h5", 
                 label_encoder_path="training/models/label_encoder_phoenix2.pkl"):
        """
        Initialize the gesture analyzer with model and label encoder
        
        Args:
            model_path: Path to the trained Keras model
            label_encoder_path: Path to the label encoder pickle file
        """
        self.model = tf.keras.models.load_model(model_path)
        
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
            self.classes = list(self.label_encoder.classes_)
        else:
            self.label_encoder = None
            self.classes = ["Zwei", "Vier"]  # Fallback
        
        self.confidence_threshold = 0.001

    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for predictions"""
        self.confidence_threshold = max(0.0, min(1.0, float(threshold)))

    def analyze(self, normalized_keypoints):
        """
        Analyze normalized keypoints using AI model
        
        Args:
            normalized_keypoints: Dict with 'left_hand' and 'right_hand' arrays
        
        Returns:
            Dict with prediction results:
                - 'class': Predicted gesture class
                - 'confidence': Confidence score
                - 'text': Human-readable prediction text
                - 'top_3': List of top 3 predictions with confidences
        """
        # Prepare features for model
        features = np.concatenate([
            normalized_keypoints['left_hand'],
            normalized_keypoints['right_hand']
        ])
        
        # Pad features to match expected input shape (127)
        if features.shape[0] < 127:
            features = np.pad(features, (0, 127 - features.shape[0]), mode='constant')
        
        # Reshape for model input
        features = features.reshape(1, -1)
        
        # Get model predictions
        probs = self.model.predict(features, verbose=0)[0]
        pred_class = np.argmax(probs)
        
        # Get top 3 predictions for debugging
        top_3_indices = np.argsort(probs)[-3:][::-1]
        top_3_predictions = [
            (self.classes[idx], float(probs[idx])) 
            for idx in top_3_indices
        ]
        
        # Map special class names
        class_name = self.classes[pred_class]
        if class_name == "si":
            class_name = "Stille"
        
        # Apply confidence threshold
        confidence = float(probs[pred_class])
        if confidence > self.confidence_threshold:
            prediction = {
                'class': class_name,
                'confidence': confidence,
                'text': f"{class_name} ({confidence*100:.1f}%)"
            }
        else:
            prediction = {
                'class': None,
                'confidence': confidence,
                'text': "Unsicher (zu geringe Konfidenz)"
            }
        
        return {
            'prediction': prediction,
            'top_3': top_3_predictions
        }
