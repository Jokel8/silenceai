"""
hand_keypoints_extractor.py
Unified hand keypoints extraction for training and pipeline
Uses HandDetectionProcessor from preprocessing
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.keypoint_extraction_processor import HandDetectionProcessor


class HandKeypointsExtractor:
    """
    Unified keypoint extractor that wraps HandDetectionProcessor
    Compatible with legacy extractKeypoints interface
    """

    def __init__(self, model_path="preprocessing/models/hand_landmarker.task"):
        print("Current working directory:", os.getcwd())
        self.image_extensions = {'.jpg', '.png'}
        
        # Use HandDetectionProcessor internally
        self.processor = HandDetectionProcessor(
            confidence=0.5,
            max_hands=2,
            model_path=model_path,
            draw_keypoints=False
        )

    def extractKeypoints(self, frame):
        """
        Extract hand keypoints from frame
        
        Args:
            frame: BGR image from camera/file
            
        Returns:
            dict with keys:
                - "frame": annotated frame
                - "left_hand": numpy array of 63 values (21 keypoints × 3 coordinates)
                - "right_hand": numpy array of 63 values (21 keypoints × 3 coordinates)
        """
        return self.processor.extractKeypoints(frame)

    def create_column_names(self):
        """Create CSV column names for keypoint data"""
        columns = ["image_path"]

        for i in range(21):
            columns.extend([
                f"left_hand_kp{i}_x",
                f"left_hand_kp{i}_y",
                f"left_hand_kp{i}_z"
            ])

        for i in range(21):
            columns.extend([
                f"right_hand_kp{i}_x",
                f"right_hand_kp{i}_y",
                f"right_hand_kp{i}_z"
            ])

        return columns

    def process_subfolder(self, subfolder_path):
        """
        Process all images in a subfolder and extract keypoints
        
        Args:
            subfolder_path: Path to folder containing images
            
        Returns:
            DataFrame with keypoint data or None if no valid data
        """
        print(f"Verarbeite Unterordner: {subfolder_path}")

        image_files = []
        for ext in self.image_extensions:
            image_files.extend(subfolder_path.glob(f"*{ext}"))
            image_files.extend(subfolder_path.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"Keine Bilddateien in {subfolder_path}")
            return None

        data_rows = []

        for image_file in sorted(image_files):
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            keypoints_data = self.extractKeypoints(image)
            keypoints_data["image_path"] = str(image_file)

            if not (
                np.all(keypoints_data["left_hand"] == 0)
                and np.all(keypoints_data["right_hand"] == 0)
            ):
                row = [keypoints_data["image_path"]]
                row.extend(keypoints_data["left_hand"])
                row.extend(keypoints_data["right_hand"])
                data_rows.append(row)

        if not data_rows:
            return None

        df = pd.DataFrame(
            data_rows,
            columns=self.create_column_names()
        )

        return df

    def extractFromFile(self, input_directory, output_directory):
        """
        Extract keypoints from all images in subfolders and save to CSV
        
        Args:
            input_directory: Root directory containing subfolders with images
            output_directory: Directory where CSV files will be saved
        """
        input_directory = Path(input_directory)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        subfolders = [d for d in input_directory.iterdir() if d.is_dir()]

        for subfolder in subfolders:
            df = self.process_subfolder(subfolder)
            if df is not None:
                csv_path = output_directory / f"{subfolder.name}_hand_keypoints.csv"
                df.to_csv(csv_path, index=False)
                print(f"CSV erstellt: {csv_path}")


if __name__ == "__main__":
    WORK_DIR = "training"
    os.chdir(WORK_DIR)

    input_dir = "rawData/test1"
    output_dir = "keypoints/test3"

    HandKeypointsExtractor().extractFromFile(input_dir, output_dir)
    print("Verarbeitung abgeschlossen")
