from stream_processor import StreamProcessor
import time

# Corrected filename: test_controller.py (underscore)
sp = StreamProcessor(ai_out_dir="data/raw/UNKNOWN_LABEL", ai_w=210, ai_h=300, target_fps=25.0)
sp.start(show_preview=True)
try:
    # Run time e.g. 10 seconds
    time.sleep(10.0)
finally:
    sp.stop()
    print("Stopped.")
