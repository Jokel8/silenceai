import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
import time

class HandKeypointsExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.image_extensions = {'.jpg', '.png'}
    
    def extractKeypoints(self, frame):
        """Verarbeite ein einzelnes Frame mit MediaPipe Hands"""
        # Zu RGB konvertieren
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Frame verarbeiten
        results = self.hands.process(rgb_frame)
        
        # Keypoints initialisieren
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)
        
        # Landmarken zeichnen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Frame anzeigen
        # cv2.imshow("Hand Gesture Prototype", frame)
        # if cv2.waitKey(5) & 0xFF == 27:  # ESC zum Beenden
        #     exit(0)
        # time.sleep(0.01)
        
        # Keypoints extrahieren
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness):
                
                hand_label = handedness.classification[0].label
                keypoints = []
                
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                if hand_label == 'Left':
                    left_hand = np.array(keypoints)
                else:
                    right_hand = np.array(keypoints)
        
        return {
            'frame': frame,
            'left_hand': left_hand,
            'right_hand': right_hand
        }

    def create_column_names(self):
        columns = ['image_path']
        
        # Linke Hand Keypoints
        for i in range(21):
            columns.extend([f'left_hand_kp{i}_x', f'left_hand_kp{i}_y', f'left_hand_kp{i}_z'])
        
        # Rechte Hand Keypoints
        for i in range(21):
            columns.extend([f'right_hand_kp{i}_x', f'right_hand_kp{i}_y', f'right_hand_kp{i}_z'])
        
        return columns
    
    def process_subfolder(self, subfolder_path):
        print(f"Verarbeite Unterordner: {subfolder_path}")
        
        # Alle Bilddateien im Unterordner finden
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(list(subfolder_path.glob(f'*{ext}')))
            image_files.extend(list(subfolder_path.glob(f'*{ext.upper()}')))
        
        if not image_files:
            print(f"Keine Bilddateien in {subfolder_path} gefunden")
            return None
        
        print(f"Gefunden: {len(image_files)} Bilddateien")
        
        # Daten für DataFrame sammeln
        data_rows = []
        
        for image_file in sorted(image_files):
            try:
                # Bild laden
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"Warnung: Bild {image_file} konnte nicht geladen werden")
                    return None
                
                # Frame mit der bereitgestellten Funktion verarbeiten
                keypoints_data = self.extractKeypoints(image)
                
                # Bildpfad zu den Ergebnissen hinzufügen
                keypoints_data['image_path'] = str(image_file)
                
            except Exception as e:
                print(f"Fehler beim Verarbeiten von {image_file}: {e}")
                continue
            
            if keypoints_data is not None and not (np.all(keypoints_data['left_hand'] == 0) and np.all(keypoints_data['right_hand'] == 0)):
                # Eine Zeile für das DataFrame erstellen
                row = [keypoints_data['image_path']]
                row.extend(keypoints_data['left_hand'])
                row.extend(keypoints_data['right_hand'])
                data_rows.append(row)
            else:
                print(f" - Keine Hand in {image_file} erkannt")
        
        if not data_rows:
            print(f"Keine gültigen Keypoints in {subfolder_path} extrahiert")
            return None
        
        # DataFrame erstellen
        columns = self.create_column_names()
        df = pd.DataFrame(data_rows, columns=columns)
        
        return df
    
    def extractFromFile(self, input_directory, output_directory):
        input_directory = Path(input_directory)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)    
        # Alle Unterordner finden
        subfolders = [d for d in input_directory.iterdir() 
                     if d.is_dir()]
        
        if not subfolders:
            print(f"Keine Unterordner in {input_directory} gefunden")
            return
        
        print(f"Gefundene Unterordner: {len(subfolders)}")
        
        for subfolder in subfolders:
            df = self.process_subfolder(subfolder)
            
            if df is not None:
                # CSV-Dateiname basierend auf Unterordnernamen
                csv_filename = f"{subfolder.name}_hand_keypoints.csv"
                csv_path = output_directory / csv_filename
                
                # DataFrame als CSV speichern
                df.to_csv(csv_path, index=False)
                print(f" - Es wurden {len(df)} Datensätze extrahiert.")
                print(f" - CSV-Datei erstellt: {csv_path}\n")
            else:
                print(f"Überspringe {subfolder.name} (keine gültigen Daten)\n")

if __name__ == "__main__":
    WORK_DIR = "SilenceAI/training"
    os.chdir(WORK_DIR)
    
    input_dir = "rawData/test1"
    output_dir = "keypoints/test3"
    
    #input_dir = "rawData/basicZweiVier"
    #output_dir = "keypoints/basicZweiVier"

    HandKeypointsExtractor().extractFromFile(input_dir, output_dir)
    print("\nVerarbeitung abgeschlossen!")