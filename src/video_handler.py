import cv2
import threading
import time
from .utils import draw_detections

class VideoHandler:
    def __init__(self, face_detector, face_tracker):
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        
        self.stream = None
        self.stop_event = None
        self.thread_read = None
        self.thread_detect = None
        
        self.latest_frame = [None]
        self.frame_lock = threading.Lock()
        self.latest_result = [None]
        self.result_lock = threading.Lock()
        
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
    
    def start_stream(self, stream_source):
        """Start video stream and processing threads."""
        self.stop_event = threading.Event()
        self.stream = cv2.VideoCapture(stream_source)
        
        if not self.stream.isOpened():
            print(f"Cannot open stream {stream_source}")
            return False
        
        self.thread_read = threading.Thread(
            target=self._read_frames,
            args=(self.stream, self.latest_frame, self.frame_lock, self.stop_event)
        )
        self.thread_read.start()
        
        self.thread_detect = threading.Thread(
            target=self._detect_faces,
            args=(self.latest_frame, self.frame_lock, self.latest_result, self.result_lock, self.stop_event)
        )
        self.thread_detect.start()
        
        self.start_time = time.time()
        self.frame_count = 0
        return True
    
    def stop_stream(self):
        """Stop video stream and processing threads."""
        if self.stop_event:
            self.stop_event.set()
        
        if self.thread_read:
            self.thread_read.join()
        
        if self.thread_detect:
            self.thread_detect.join()
        
        if self.stream:
            self.stream.release()
        
        cv2.destroyAllWindows()
        
        self.stop_event = None
        self.thread_read = None
        self.thread_detect = None
        self.stream = None
    
    def _read_frames(self, stream, latest_frame, frame_lock, stop_event):
        """Thread function for reading frames from the video stream."""
        while not stop_event.is_set():
            ret, frame = stream.read()
            if not ret:
                break
            with frame_lock:
                latest_frame[0] = frame.copy()
    
    def _detect_faces(self, latest_frame, frame_lock, latest_result, result_lock, stop_event):
        """Thread function for detecting faces in frames."""
        while not stop_event.is_set():
            with frame_lock:
                if latest_frame[0] is None:
                    continue
                frame = latest_frame[0].copy()
            
            detections = self.face_detector.detect_and_recognize(frame)
            tracked_faces = self.face_tracker.update(detections)
            
            with result_lock:
                latest_result[0] = (frame, tracked_faces)
    
    def get_processed_frame(self):
        """Get the latest processed frame with detections."""
        with self.result_lock:
            if self.latest_result[0] is not None:
                frame, tracked_faces = self.latest_result[0]
                self.latest_result[0] = None
                
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                frame_with_detections = draw_detections(frame.copy(), tracked_faces)
                return frame_with_detections, self.fps, self.face_tracker.get_statistics()
        
        return None, self.fps, None