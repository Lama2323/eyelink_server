import cv2
import threading
import time
from core.face_detection import FaceDetector
from core.face_recognition import FaceRecognitionSystem
from core.face_tracking import FaceTracker

class CameraStream:
    def __init__(self, stream_source, camera_id):
        self.stream_source = stream_source
        self.camera_id = camera_id
        self.stream = None
        self.stop_event = threading.Event()
        self.latest_frame = [None]
        self.frame_lock = threading.Lock()
        self.latest_result = [None]
        self.result_lock = threading.Lock()
        
        self.detector = FaceDetector()
        self.recognizer = FaceRecognitionSystem()
        self.tracker = FaceTracker()
        
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
    
    def start(self):
        self.stream = cv2.VideoCapture(self.stream_source)
        if not self.stream.isOpened():
            return False
        
        self.stop_event.clear()
        threading.Thread(target=self._read_frames).start()
        threading.Thread(target=self._process_frames).start()
        return True
    
    def stop(self):
        self.stop_event.set()
        if self.stream:
            self.stream.release()
    
    def _read_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.stream.read()
            if not ret:
                break
            with self.frame_lock:
                self.latest_frame[0] = frame.copy()
    
    def _process_frames(self):
        while not self.stop_event.is_set():
            with self.frame_lock:
                if self.latest_frame[0] is None:
                    continue
                frame = self.latest_frame[0].copy()
            
            # Detect and process faces
            detections = self.detector.detect_faces(frame)
            recognition_results = []
            
            for detection in detections:
                aligned_face = self.recognizer.align_face(frame, detection['landmarks'])
                embedding = self.recognizer.get_face_embedding(aligned_face)
                
                # Compare with known faces
                if self.recognizer.index is not None:
                    D, I = self.recognizer.index.search(embedding.reshape(1, -1), k=1)
                    distance = D[0][0]
                    idx = I[0][0]
                    
                    if distance < 1.05:  # threshold
                        recognition_results.append({
                            'name': self.recognizer.known_names[idx],
                            'recognized': True
                        })
                    else:
                        recognition_results.append({
                            'name': 'unknown',
                            'recognized': False
                        })
                else:
                    recognition_results.append({
                        'name': 'unknown',
                        'recognized': False
                    })
            
            # Update tracking
            tracked_faces = self.tracker.update_tracks(detections, recognition_results)
            
            with self.result_lock:
                self.latest_result[0] = (frame, tracked_faces)
            
            # Update FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()