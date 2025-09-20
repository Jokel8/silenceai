import os
import math
import cv2 # See video 1 for installation
import numpy as np

# Replace "0" with a file path to work with a saved video
stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("No stream :(")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))

# handle invalid fps returned by some webcams
if fps is None or math.isnan(fps) or fps <= 0:
    print("Warning: stream FPS invalid, falling back to 30")
    fps = 30.0
out_fps = int(round(fps))

out_path = "Wettbewerbe/BWKI-2025/assets/sharpening.mp4"
out_dir = os.path.dirname(out_path)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# list of FourCC video codes: https://softron.zendesk.com/hc/en-us/articles/207695697-List-of-FourCC-codes-for-video-codecs
fourcc_candidates = [cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                     cv2.VideoWriter_fourcc(*'mp4v'),
                     cv2.VideoWriter_fourcc(*'XVID'),
                     cv2.VideoWriter_fourcc(*'MJPG')]

output = None
for fourcc in fourcc_candidates:
    vw = cv2.VideoWriter(out_path, fourcc, out_fps, frameSize=(width, height))
    if vw.isOpened():
        output = vw
        break

if output is None:
    print("Warning: VideoWriter could not be opened. Continuing without saving output file.")

try:
    while True:
        ret, frame = stream.read()
        if not ret: # if no frames are returned
            print("No more stream :(")
            break
        
        # only resize if we have valid dims
        if width > 0 and height > 0:
            frame = cv2.resize(frame, (width, height))
        sharpen_filter = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        frame = cv2.filter2D(frame, ddepth=-1, 
            kernel=sharpen_filter)
        if output is not None:
            output.write(frame)
        cv2.imshow("Webcam!", frame)
        if cv2.waitKey(1) == ord('q'): # press "q" to quit
            break
finally:
    stream.release()
    if output is not None:
        output.release()
    cv2.destroyAllWindows() #!