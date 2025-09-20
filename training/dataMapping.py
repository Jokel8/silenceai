import os
import shutil

WORK_DIR = "SilenceAI/training/rawData/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/"
os.chdir(WORK_DIR)
    
training_classes_file = "annotations/automatic/trainingClasses.txt"
alignment_file = "annotations/automatic/train.alignment"
output_dir = "../../../mapped_phoenix"

# 1. Mapping classlabel -> signstate erstellen
class_map = {}
with open(training_classes_file, "r", encoding="utf-8") as f:
    next(f)  # erste Zeile überspringen (Header)
    for line in f:
        signstate, classlabel = line.strip().split()
        class_map[int(classlabel)] = signstate

# 2. Alignment-Datei durchgehen
with open(alignment_file, "r", encoding="utf-8") as f:
    for line in f:
        img_path, classlabel = line.strip().split()
        classlabel = int(classlabel)

        # 3. Signstate herausfinden
        signstate = class_map[classlabel]

        # letzte Ziffer entfernen → Sign ohne Zustandsnummer
        sign_name = signstate.rstrip("0123456789")

        # Zielordner
        target_dir = os.path.join(output_dir, sign_name)
        os.makedirs(target_dir, exist_ok=True)

        # 4. Bild kopieren
        target_path = os.path.join(target_dir, os.path.basename(img_path))
        shutil.copy(img_path, target_path)
