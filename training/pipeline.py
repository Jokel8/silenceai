import cv2
import os
import time
import numpy as np
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import mediapipe as mp
import tensorflow as tf
import joblib

from extractKeypoints import HandKeypointsExtractor
from normalizKeypoints import HandKeypointsNormalizer

handKeypointExtractor = HandKeypointsExtractor()
handKeypointsNormalizer = HandKeypointsNormalizer()

# Model and LabelEncoder paths
MODEL_PATH = "silenceai/Training/models/gesture_model_phoenix1.h5"
LABEL_ENCODER_PATH = "silenceai/Training/models/label_encoder_phoenix.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    CLASSES = list(label_encoder.classes_)
else:
    CLASSES = ["Zwei", "Vier"]  # Fallback

def capture_camera_frames():
    """Step 1: Capture frames from camera at 25 FPS"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 25)
    
    frame_time = 1/25
    prev_time = time.time()
    
    while cap.isOpened():
        curr_time = time.time()
        if curr_time - prev_time < frame_time:
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Add FPS display
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        yield frame
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def analyze_keypoints(normalized_data):
    """Step 4: Analyze normalized keypoints using AI model"""
    
    # Prepare features for model
    features = np.concatenate([
        normalized_data['left_hand'],
        normalized_data['right_hand']
    ])
    
    # Pad features to match expected input shape (127)
    if features.shape[0] < 127:
        features = np.pad(features, (0, 127 - features.shape[0]), mode='constant')
    
    # Reshape for model input
    features = features.reshape(1, -1)
    
    # Get model predictions
    probs = model.predict(features, verbose=0)[0]
    pred_class = np.argmax(probs)
    
    # Get top 3 predictions for debugging
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top_3_predictions = [
        (CLASSES[idx], probs[idx]) 
        for idx in top_3_indices
    ]
    
    # Apply confidence threshold
    confidence_threshold = 0.001
    if probs[pred_class] > confidence_threshold:
        prediction = {
            'class': CLASSES[pred_class],
            'confidence': probs[pred_class],
            'text': f"Zeichen: {CLASSES[pred_class]} ({probs[pred_class]*100:.1f}%)"
        }
    else:
        prediction = {
            'class': None,
            'confidence': probs[pred_class],
            'text': "Unsicher (zu geringe Konfidenz)"
        }
    
    return {
        **normalized_data,  # Include previous data
        'prediction': prediction,
        'top_3': top_3_predictions
    }

def run_pipeline():
    """Main pipeline that executes all steps in sequence"""
    print("Starting pipeline...")
    
    while True:
        # Step 1: Get frame from camera
        for frame in capture_camera_frames():
            # Step 2: Extract keypoints
            keypoints = handKeypointExtractor.extractKeypoints(frame)
            frame = keypoints.pop('frame')
            
            # Step 3: Normalize keypoints
            keypoints = handKeypointsNormalizer.relative_to_wrist_normalize(keypoints)
            keypoints = handKeypointsNormalizer.global_minmax_normalize(keypoints)
            
            # Step 4: Analyze with AI model
            analysis_result = analyze_keypoints(keypoints)
            
            # Display results
            cv2.putText(frame, analysis_result['prediction']['text'], 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Pipeline Output", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                exit(0)
            
            # Print debugging info
            if analysis_result['prediction']['class'] is not None:
                print("\nTop 3 Predictions:")
                for class_name, prob in analysis_result['top_3']:
                    print(f"Class {class_name}: {prob*100:.2f}%")
            else:
                print("No confident prediction.")
            
            # Return result for further processing
            yield analysis_result

if __name__ == "__main__":
    # Run the pipeline
    for result in run_pipeline():
        # Here you can add additional processing steps
        # Each iteration provides:
        # - result['frame']: processed image
        # - result['left_hand']: normalized keypoints for left hand
        # - result['right_hand']: normalized keypoints for right hand
        # - result['prediction']: AI model prediction
        # - result['top_3']: top 3 predictions with confidences
        pass