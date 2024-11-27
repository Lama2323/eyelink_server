import cv2
import numpy as np
from config.settings import *

class FaceDetector:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="",
            input_size=FACE_DETECTION_SIZE,
            score_threshold=FACE_DETECTION_SCORE_THRESHOLD,
            nms_threshold=FACE_DETECTION_NMS_THRESHOLD,
            top_k=FACE_DETECTION_TOP_K
        )
    
    def detect_faces(self, frame):
        small_frame = cv2.resize(frame, FACE_DETECTION_SIZE)
        height, width, _ = small_frame.shape
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(small_frame)
        
        if faces is None:
            return []
            
        scale_x = frame.shape[1] / width
        scale_y = frame.shape[0] / height
        
        detections = []
        for face in faces:
            bbox = face[:4] * [scale_x, scale_y, scale_x, scale_y]
            landmarks = face[4:14].reshape((5, 2))
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y
            
            detections.append({
                'bbox': bbox.astype(np.int32),
                'landmarks': landmarks
            })
            
        return detections