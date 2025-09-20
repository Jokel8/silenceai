import os
import random
import shutil
from glob import glob
from PIL import Image

# Simple dataset builder: reads RGBA PNGs from data/raw/<label> and
# creates train/val/test splits in data/{train,val,test}/<label>.

SRC_ROOT = "data/raw"
DST_ROOT = "data"
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
TARGET_SIZE = (210, 300)  # width, height

os.makedirs(DST_ROOT, exist_ok=True)

labels = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))]
if not labels:
    print("No labels found in data/raw. Create folders like data/raw/label1 and put PNGs there.")
    exit(1)

for lbl in labels:
    paths = glob(os.path.join(SRC_ROOT, lbl, "*.png"))
    if not paths:
        print(f"No images for label {lbl}, skipping")
        continue
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train+n_val]
    test_paths = paths[n_train+n_val:]

    for split, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        out_dir = os.path.join(DST_ROOT, split, lbl)
        os.makedirs(out_dir, exist_ok=True)
        for i, p in enumerate(split_paths):
            try:
                im = Image.open(p)
            except Exception as e:
                print(f"Failed to open {p}: {e}")
                continue
            # Ensure RGBA or convert
            if im.mode not in ("RGBA", "RGB"):
                im = im.convert("RGBA")
            # If alpha present, composite over white
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255,255,255))
                bg.paste(im, mask=im.split()[3])
                im = bg
            # resize to TARGET_SIZE
            im = im.resize(TARGET_SIZE, Image.LANCZOS)
            dst = os.path.join(out_dir, f"{i:06d}.png")
            im.save(dst)

print("Dataset prepared: data/train, data/val, data/test")
