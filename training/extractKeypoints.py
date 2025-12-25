import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class HandKeypointsExtractor:

    def __init__(self, model_path="training/models/hand_landmarker.task"):
        print("Current working directory:", os.getcwd())
        self.image_extensions = {'.jpg', '.png'}

        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=model_path
            ),
            num_hands=2
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extractKeypoints(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        if result.hand_landmarks and result.handedness:
            h, w, _ = frame.shape

            for hand_landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness
            ):
                label = handedness[0].category_name
                keypoints = []

                for lm in hand_landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])

                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                if label == "Left":
                    left_hand = np.array(keypoints)
                elif label == "Right":
                    right_hand = np.array(keypoints)

        return {
            "frame": frame,
            "left_hand": left_hand,
            "right_hand": right_hand
        }

    def create_column_names(self):
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