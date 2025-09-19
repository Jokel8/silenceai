import pandas as pd
import numpy as np
from pathlib import Path
import os

class HandKeypointsNormalizer:
    def __init__(self, input_directory, output_directory):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def relative_to_wrist_normalize(self, df):
        df_norm = df.copy()
        for hand in ['left_hand', 'right_hand']:
            wrist_cols = [f'{hand}_kp0_x', f'{hand}_kp0_y', f'{hand}_kp0_z']
            if all(col in df.columns for col in wrist_cols):
                wrist_x, wrist_y, wrist_z = df[wrist_cols].values.T
                for kp in range(21):
                    x_col, y_col, z_col = f'{hand}_kp{kp}_x', f'{hand}_kp{kp}_y', f'{hand}_kp{kp}_z'
                    if x_col in df.columns:
                        df_norm[x_col] = df[x_col] - wrist_x
                        df_norm[y_col] = df[y_col] - wrist_y
                        df_norm[z_col] = df[z_col] - wrist_z
        return df_norm
    
    def global_minmax_normalize(self, df):
        df_norm = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        max_abs_value = df[numeric_cols].abs().max().max()
        if max_abs_value > 0:
            df_norm[numeric_cols] = df[numeric_cols] / max_abs_value
        return df_norm
    
    def process_file(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Beide Normalisierungen anwenden
        df = self.relative_to_wrist_normalize(df)
        df = self.global_minmax_normalize(df)
        
        # Nur Keypoint-Spalten (ohne image_path)
        keypoint_cols = [col for col in df.columns if col != 'image_path']
        final_df = df[keypoint_cols]
        
        # Ausgabe speichern
        output_file = self.output_directory / f"{csv_file.stem}_normalized.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Verarbeitet: {csv_file.name} -> {output_file.name}")
    
    def normalize(self):
        csv_files = list(self.input_directory.glob("*_hand_keypoints.csv"))
        if not csv_files:
            print("Keine Keypoint-Dateien gefunden")
            return
        
        for csv_file in csv_files:
            self.process_file(csv_file)
        
        print(f"Fertig! {len(csv_files)} Dateien verarbeitet")

if __name__ == "__main__":
    WORK_DIR = "SilenceAI/training"
    os.chdir(WORK_DIR)
    
    input_dir = "keypoints/basicZweiVier"
    output_dir = "datasets/basicZweiVier"
    normalizer = HandKeypointsNormalizer(input_dir, output_dir).normalize()