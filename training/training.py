import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

class HandGestureClassifier:
    def __init__(self, identifier="test"):
        self.data_directory = Path("datasets/" + identifier + "/")
        self.model = None
        self.label_encoder = LabelEncoder()
        self.log_dir = f"logs/{identifier}/"
        
    def load_and_prepare_data(self):
        csv_files = list(self.data_directory.glob("*_dataset.csv"))
        if not csv_files:
            raise ValueError("Keine normalisierten CSV-Dateien gefunden")
        
        all_data = []
        all_labels = []
        
        for csv_file in csv_files:
            # Label aus Dateiname extrahieren (vor "_dataset.csv")
            label = csv_file.stem.split('_')[0]
            
            df = pd.read_csv(csv_file)
            
            # Alle Frames verwenden
            keypoint_cols = [col for col in df.columns if col != 'id']
            
            for _, row in df.iterrows():
                all_data.append(row[keypoint_cols].values.astype(np.float32))
                all_labels.append(label)
        
        # Zu numpy arrays konvertieren
        X = np.array(all_data)
        y = np.array(all_labels)
        
        # Labels encodieren
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Daten geladen: {len(X)} Samples, {X.shape[1]} Features")
        print(f"Klassen: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def create_model(self, input_dim, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_callbacks(self):
        # TensorBoard Callback
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, 
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Model Checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/gesture_model_test2.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        return [tensorboard_cb, early_stopping, checkpoint]
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        # Daten laden und vorbereiten
        X, y = self.load_and_prepare_data()
        
        # Train/Validation Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Modell erstellen
        num_classes = len(np.unique(y))
        self.model = self.create_model(X.shape[1], num_classes)
        
        print("Modell-Architektur:")
        self.model.summary()
        
        # Callbacks erstellen
        callbacks = self.create_callbacks()
        
        # Training
        print("\\nStarte Training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Finale Evaluierung
        test_loss, test_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\\nFinale Metriken:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\\nTensorBoard Logs: {self.log_dir}")
        print("Starte TensorBoard mit: tensorboard --logdir=logs")
        
        return history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Modell muss erst trainiert werden")
        
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        return predicted_labels, predictions
    
    def save_model(self, filepath="hand_gesture_model.h5"):
        if self.model is not None:
            self.model.save(filepath)
            print(f"Modell gespeichert: {filepath}")
    
    def load_model(self, filepath="hand_gesture_model.h5"):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Modell geladen: {filepath}")

def main():
    # Classifier initialisieren
    classifier = HandGestureClassifier("test2")
    
    # Modell trainieren
    history = classifier.train(
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    # Modell speichern
    classifier.save_model("models/gesture_model_test2.h5")
    
    print("\\nTraining abgeschlossen!")
    print("Öffne TensorBoard mit: tensorboard --logdir=logs")

if __name__ == "__main__":
    WORK_DIR = "SilenceAI/training"
    os.chdir(WORK_DIR)
    
    # TensorFlow Konfiguration
    tf.random.set_seed(42)
    
    # GPU falls verfügbar
    if tf.config.list_physical_devices('GPU'):
        print("GPU verfügbar - verwende GPU für Training")
    else:
        print("Verwende CPU für Training")
    
    main()