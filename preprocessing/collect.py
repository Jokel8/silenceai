import os
import time
import argparse
import threading
from pathlib import Path
import cv2

from stream_processor import StreamProcessor

"""
Simple collection utility that uses StreamProcessor to save AI RGBA crops into
data/raw/<label>/ as PNGs. Uses the on_ai_frame hook for low-latency saving.

Usage:
  python preprocessing/collect.py --label YOUR_LABEL --out data/raw --max 500

If --max is not given, run until Ctrl-C.
"""


def make_saver(out_dir: Path, label: str):
    lbl_dir = out_dir / label
    lbl_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    lock = threading.Lock()

    def on_frame(rgba):
        nonlocal counter
        try:
            # write without blocking main thread: use cv2.imencode + tofile for Windows-safe write
            p = lbl_dir / f"frame_{counter:06d}.png"
            # cv2.imwrite can be used directly; keep it simple
            cv2.imwrite(str(p), rgba)
            with lock:
                counter += 1
        except Exception as e:
            print(f"Failed to save frame: {e}")

    return on_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--out', default='data/raw')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--max', type=int, default=0, help='max frames to collect; 0=unlimited')
    p.add_argument('--preview', action='store_true', help='show preview windows')
    args = p.parse_args()

    out_dir = Path(args.out)
    saver = make_saver(out_dir, args.label)

    sp = StreamProcessor(camera_index=args.camera, ai_out_dir=str(out_dir / args.label), on_ai_frame=saver)
    sp.start(show_preview=args.preview)
    print(f"Collecting for label '{args.label}' into {out_dir / args.label}. Press Ctrl-C to stop.")
    try:
        if args.max > 0:
            # wait until saver counter reaches max
            while True:
                time.sleep(0.5)
                # check files
                n = len(list((out_dir / args.label).glob('*.png')))
                if n >= args.max:
                    print(f"Reached {n} frames, stopping.")
                    break
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sp.stop()
        print("Stopped StreamProcessor.")

if __name__ == '__main__':
    main()
