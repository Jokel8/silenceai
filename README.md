# silenceai
**Translator for DGS**
Gebärdensprache ist eine komplexe und ausdrucksstarke Kommunikationsform, die für Millionen Menschen weltweit essenziell ist. Dennoch gibt es in vielen Situationen Kommunikationsbarrieren zwischen Menschen, die Gebärdensprache verwenden, und jenen, die diese nicht beherrschen. Unser Projekt SilentAI entwickelt ein KI-gestütztes Übersetzungsprogramm, das Gebärdensprache in Echtzeit in gesprochene und geschriebene Sprache übersetzt.

**Unsere Ziele:**
1.	Automatische Erkennung direkt über eine Kamera
2.	Echtzeitübersetzung mit Sprachausgabe
3.	Bei genug Zeit: Benutzerfrendliche Schnittstelle als Webapp oder Android App

**Unsere Vorgehensweise:**
1.  Pre-Processing (Hintergrund entfernen, Helligkeit und Kontrast normalisieren) @Jan
2.  Keypoint-Erkennung (Skelett der Hände berechnen) mit MediaPipe Holistic @Jonas
3.  Gebärdenerkennung durch ein eigenes Modell mit den RWTH Datensätzen @Jonas
4.  Post-Processing (Grammatikalische Korrektur und ggf. einfügen von Präpositionen) mit Qwen2 0.5B Parameter @Kende


---

## Quick start — Setup
1. Install Python 3.8+ (3.10/3.11 recommended) and Git.
2. Create and activate a virtual environment (recommended):
   - PowerShell: python -m venv .venv; .\.venv\Scripts\Activate.ps1
3. Install dependencies:
   - pip install -r requirements.txt
4. Optional: set CAMERA_INDEX environment variable to change default camera index:
   - PowerShell: $env:CAMERA_INDEX=1

## Workflow overview
The repository contains three main stages: data collection, preprocessing (make dataset), and training. A small Tkinter GUI is provided to launch training and monitor logs.

1) Collect data (live)
- Use the data collection tool to capture labeled frames or short clips from a webcam.
- Example (run from repository root):
  - python -m preprocessing.collect --help  # see available options
  - python -m preprocessing.collect --label LEFT --out data/raw/LEFT
Notes:
- The camera index can be set with `--camera` or via the `CAMERA_INDEX` environment variable.
- The collector saves images/videos under `data/raw/<LABEL>` by default.

2) Preprocess / Create dataset
- Prepare a processed dataset suitable for training using the provided script.
- Example:
  - python -m preprocessing.make_dataset --help
  - python -m preprocessing.make_dataset --raw data/raw --out data/processed
This step will:
- Convert images to RGB (drop alpha if present),
- Resize to the configured model size,
- Split data into train/validation/test folders.

3) Train a model
- CLI training (recommended for reproducibility):
  - python -m training.train --help
  - Example run:
    python -m training.train --data data/processed --model output/models/model.h5 --checkpoint output/checkpoints --epochs 50 --batch-size 32
Features implemented in the training script:
- ModelCheckpoint (best checkpoint saved),
- EarlyStopping,
- TensorBoard logging,
- Automatic export of the best checkpoint to `output/models/best_model.h5` and SavedModel format,
- Optional conversion to TFLite (float16) via CLI flags.

4) GUI (optional)
- A simple training GUI is available at `gui/train_gui.py`.
- Run on Windows (PowerShell):
  - .\gui\run_train_gui.ps1
  or
  - python -m gui.train_gui
- GUI features:
  - Select processed data folder, output model path and checkpoint folder,
  - Set epochs and batch size,
  - Start/Stop training (runs the training script as subprocess),
  - Live log display and save/load settings,
  - Open model folder in Explorer.

5) Inference / Interpreter
- Use `interpreter.py` as a simple wrapper for loading the trained model and running inference on keypoints or frames.
- Example usage is provided in `Interpreter-V1.py` shim (now `interpreter.py`). See the file header for import examples.

## Recommended quick workflow for a friend
1. Clone the repo and set up the virtualenv, install requirements.
2. Collect a small labeled dataset using the collector for a few gestures (e.g. 100–500 images per class).
3. Run `make_dataset` to produce `data/processed`.
4. Start training with the CLI or the GUI. Use small epochs first to validate everything.
5. After training completes, the best model will be available in `output/models/`.
6. Run `python -m interpreter` or import `interpreter` from another script to load the model and run predictions.

## Troubleshooting
- Camera not opening: try setting the camera index (`--camera` or CAMERA_INDEX env) or install OpenCV with `pip install opencv-python-headless` if GUI not needed.
- MediaPipe issues: ensure compatible versions from `requirements.txt` are installed.
- If the GUI won't start: run `python -m gui.train_gui` from the repository root and check printed errors.

## Next steps / Wishlist
- Add GUI button to export best checkpoint (planned).
- Auto-launch TensorBoard from GUI (planned).
- Add example notebook for inference and CI tests.

---

If you need a one-page quick-help for your friend, copy the "Recommended quick workflow" section and paste it into a message.
