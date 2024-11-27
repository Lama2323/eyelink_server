import cv2
import threading
import time
from src.face.tracking import TrackedFace
from src.camera.detection import detect_faces 

def read_frames(stream, latest_frame, frame_lock, stop_event):
    """Thread function to read frames from camera"""
    while not stop_event.is_set():
        ret, frame = stream.read()
        if not ret:
            break
        with frame_lock:
            latest_frame[0] = frame.copy()

class CameraStream:
    """Class to manage camera streams"""
    def __init__(self, stream_source, camera_id):
        self.stream_source = stream_source
        self.camera_id = camera_id
        self.stream = None
        self.stop_event = threading.Event()
        self.thread_read = None
        self.thread_detect = None
        self.latest_frame = [None]
        self.frame_lock = threading.Lock()
        self.latest_result = [None]
        self.result_lock = threading.Lock()
        self.tracked_faces = {}
        self.face_id_counter = 0
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.init_complete = threading.Event()
        
        # Initialize models
        self.yunet = cv2.FaceDetectorYN.create(
            model="model/yunet.onnx",
            config="",
            input_size=(160, 160),
            score_threshold=0.6,
            nms_threshold=0.4,
            top_k=5000
        )
        self.recognizer_net = cv2.dnn.readNetFromONNX('model/mobilefacenet.onnx')
    
    def start(self):
        """Start camera stream and detection threads"""
        self.stream = cv2.VideoCapture(self.stream_source)
        if not self.stream.isOpened():
            print(f"Cannot open stream {self.stream_source}")
            return False
        
        self.stop_event.clear()
        self.thread_read = threading.Thread(
            target=read_frames,
            args=(self.stream, self.latest_frame, self.frame_lock, self.stop_event)
        )
        self.thread_detect = threading.Thread(
            target=detect_faces,  # This was causing the error
            args=(self.latest_frame, self.frame_lock, self.latest_result, 
                  self.result_lock, self.stop_event, self.yunet, self.recognizer_net)
        )
        
        self.thread_read.start()
        self.init_complete.set()
        self.thread_detect.start()
        return True
    
    def stop(self):
        """Stop camera stream and detection threads"""
        if self.stop_event:
            self.stop_event.set()
        if self.thread_read:
            self.thread_read.join()
        if self.thread_detect:
            self.thread_detect.join()
        if self.stream:
            self.stream.release()
        
        self.stop_event = threading.Event()
        self.thread_read = None
        self.thread_detect = None
        self.stream = None