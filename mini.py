import cv2
import os
import time
import numpy as np
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import mediapipe as mp
import tensorflow as tf
import joblib
import threading
import pyttsx3

from training.extractKeypoints import HandKeypointsExtractor
from training.normalizKeypoints import HandKeypointsNormalizer
from preprocessing.keypoint_normalization_processor import KeypointsNormalizeProcessor
from coreprocessing.gesture_analyzer import GestureAnalyzer
from ui.graphics.keypoint_drawer import KeypointDrawerProcessor

speech_lock = threading.Lock()

handKeypointExtractor = HandKeypointsExtractor()
handKeypointsNormalizer = HandKeypointsNormalizer()
keypointNormalizer = KeypointsNormalizeProcessor()
gestureAnalyzer = GestureAnalyzer()
keypointDrawer = KeypointDrawerProcessor(keypoint_size=5, line_thickness=2, color=(255, 255, 0))

# Model and LabelEncoder paths
MODEL_PATH = "training/models/gesture_model_phoenix2.h5"
LABEL_ENCODER_PATH = "training/models/label_encoder_phoenix2.pkl"

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
    analysis_result = gestureAnalyzer.analyze(normalized_data)
    return {
        **normalized_data,  # Include previous data
        **analysis_result   # Include prediction and top_3
    }

def say_text(text):
    with speech_lock:
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        engine.say(text)
        engine.runAndWait()

def run_pipeline():
    """Main pipeline that executes all steps in sequence"""
    print("Starting pipeline...")
    
    while True:
        # Step 1: Get frame from camera
        for frame in capture_camera_frames():
            # Step 2: Extract keypoints
            keypoints = handKeypointExtractor.extractKeypoints(frame)
            frame = keypoints.pop('frame')
            
            # Step 2.5: Draw keypoints on frame
            frame = keypointDrawer.process(frame, keypoints)
            
            # Step 3: Normalize keypoints
            keypoints = keypointNormalizer.relative_to_wrist_normalize(keypoints)
            keypoints = keypointNormalizer.global_minmax_normalize(keypoints)
            
            # Step 4: Analyze with AI model
            analysis_result = analyze_keypoints(keypoints)
            
            # Display results
            prediction = str(analysis_result['prediction']['text'])
            cv2.putText(frame, prediction, 
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
                
            if analysis_result['prediction']['confidence'] > 0.01:
                if speech_lock.acquire(blocking=False):
                    try:
                        thread = threading.Thread(target=say_text, args=(prediction,))
                        thread.start()
                    finally:
                        speech_lock.release()
            
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