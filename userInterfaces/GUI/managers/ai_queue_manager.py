"""
ai_queue_manager.py
Manages the AI processing queue and worker thread
"""

import cv2
import queue
import threading


class AIQueueManager:
    """Manages AI frame queue and processing"""
    
    def __init__(self, max_size=128):
        self.ai_q = queue.Queue(maxsize=max_size)
        self._worker = None
        self._stop_event = None
    
    def start_worker(self, stop_event):
        """Start the AI worker thread"""
        self._stop_event = stop_event
        self._worker = threading.Thread(
            target=self._ai_saver_worker,
            args=(stop_event,),
            daemon=True
        )
        self._worker.start()
    
    def stop_worker(self):
        """Stop the AI worker thread"""
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
    
    def push_frame(self, rgba_frame):
        """
        Push a frame to the AI queue
        
        Args:
            rgba_frame: RGBA image to push
            
        Returns:
            bool: True if pushed successfully, False if queue full
        """
        try:
            self.ai_q.put_nowait(rgba_frame)
            return True
        except queue.Full:
            return False
    
    def _ai_saver_worker(self, stop_event):
        """Worker thread that processes AI queue"""
        idx = 0
        while not stop_event.is_set() or not self.ai_q.empty():
            try:
                rgba = self.ai_q.get(timeout=0.2)
            except queue.Empty:
                continue
            
            ai_output = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
            # This is where you would save or process the AI output
            # For now, we just convert it
            yield ai_output
            idx += 1
    
    def get_queue_size(self):
        """Get current queue size"""
        return self.ai_q.qsize()
    
    def is_queue_full(self):
        """Check if queue is full"""
        return self.ai_q.full()
