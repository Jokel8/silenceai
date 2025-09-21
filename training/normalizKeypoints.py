import pandas as pd
import numpy as np
from pathlib import Path
import os

class HandKeypointsNormalizer:    
    def relative_to_wrist_normalize(self, points_dict):
        """
        Normalize keypoints relative to wrist position for both hands
        Args:
            points_dict: Dictionary with 'left_hand' and 'right_hand' arrays
        Returns:
            Dictionary with normalized arrays
        """
        normalized = {}
        for hand in ['left_hand', 'right_hand']:
            points_array = points_dict[hand]
            if np.any(points_array != 0):
                # Reshape to (21,3) for easier processing
                points = points_array.reshape(21, 3)
                # First point is wrist
                wrist = points[0]
                # Normalize relative to wrist
                normalized[hand] = (points - wrist).flatten()
            else:
                normalized[hand] = points_array
        return normalized

    def global_minmax_normalize(self, points_dict):
        """
        Perform global min-max normalization on both hands
        Args:
            points_dict: Dictionary with 'left_hand' and 'right_hand' arrays
        Returns:
            Dictionary with normalized arrays
        """
        normalized = {}
        for hand in ['left_hand', 'right_hand']:
            points_array = points_dict[hand]
            max_abs_value = np.abs(points_array).max()
            if max_abs_value > 0:
                normalized[hand] = points_array / max_abs_value
            else:
                normalized[hand] = points_array
        return normalized
    
    def process_file(self, csv_file, output_directory):
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame to NumPy arrays for each hand
        hands_data = {
            'left_hand': np.zeros((21, 3)),
            'right_hand': np.zeros((21, 3))
        }
        
        for hand in ['left_hand', 'right_hand']:
            hand_cols = [f'{hand}_kp{i}_{xyz}' for i in range(21) for xyz in ['x','y','z']]
            if all(col in df.columns for col in hand_cols):
                hand_values = df[hand_cols].values
                hands_data[hand] = hand_values.reshape(-1, 21, 3)
        
        # Apply normalizations
        for hand in hands_data:
            if np.any(hands_data[hand] != 0):
                hands_data[hand] = self.relative_to_wrist_normalize(hands_data[hand])
                hands_data[hand] = self.global_minmax_normalize(hands_data[hand])
        
        # Convert back to DataFrame format
        for hand in ['left_hand', 'right_hand']:
            flat_data = hands_data[hand].reshape(-1, 63)
            for i in range(21):
                for j, xyz in enumerate(['x','y','z']):
                    col = f'{hand}_kp{i}_{xyz}'
                    df[col] = flat_data[:, i*3 + j]
    
        # Dataset-IDs basierend auf Dateiname vor .avi erstellen
        try:
            df['dataset_name'] = df['image_path'].str.extract(r'([^\\]+)\.avi')[0]
            dataset_mapping = {name: idx+1 for idx, name in enumerate(df['dataset_name'].unique())}
            df['dataset_id'] = df['dataset_name'].map(dataset_mapping)

            # Frame-IDs innerhalb jedes Datasets extrahieren und hochzählen
            df['frame_number'] = df['image_path'].str.extract(r'fn(\d+)')[0].astype(int)
            df['frame_id'] = df.groupby('dataset_name')['frame_number'].rank(method='dense').astype(int)

            # Kombinierte ID erstellen (dataset.frame)
            df['id'] = df['dataset_id'].astype(str) + '.' + df['frame_id'].astype(str)
            
            # Nur ID und Keypoint-Spalten (ohne image_path)
            keypoint_cols = [col for col in df.columns if col not in ['image_path', 'dataset_name', 'dataset_id', 'frame_number', 'frame_id']]
            final_df = df[['id'] + keypoint_cols]
            
        except Exception as e:
            print("Es handelt sich nicht um die Standardbezeichnung für Phoenix-Daten", e)
            # Nur Keypoint-Spalten (ohne image_path) 
            keypoint_cols = [col for col in df.columns if col != 'image_path']
            final_df = df[keypoint_cols]
            
        
        # Ausgabe speichern
        clean_stem = csv_file.stem.replace("_hand_keypoints", "")
        output_file = output_directory / f"{clean_stem}_dataset.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Verarbeitet: {csv_file.name} -> {output_file.name}")
    
    def normalizeFromFile(self, input_directory, output_directory):
        input_directory = Path(input_directory)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(input_directory.glob("*_hand_keypoints.csv"))
        if not csv_files:
            print("Keine Keypoint-Dateien gefunden")
            return
        
        for csv_file in csv_files:
            self.process_file(csv_file, output_directory)
        
        print(f"Fertig! {len(csv_files)} Dateien verarbeitet")

if __name__ == "__main__":
    WORK_DIR = "SilenceAI/training"
    os.chdir(WORK_DIR)
    
    input_dir = "keypoints/test3"
    output_dir = "datasets/test3"
    normalizer = HandKeypointsNormalizer().normalizeFromFile(input_dir, output_dir)