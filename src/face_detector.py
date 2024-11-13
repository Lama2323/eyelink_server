import cv2
import numpy as np
import os
import faiss
from .config import *
from .utils import align_face

class FaceDetector:
    def __init__(self):
        # Initialize YuNet face detector
        self.yunet = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="",
            input_size=FACE_DETECTION_INPUT_SIZE,
            score_threshold=FACE_DETECTION_SCORE_THRESHOLD,
            nms_threshold=FACE_DETECTION_NMS_THRESHOLD,
            top_k=FACE_DETECTION_TOP_K
        )
        
        # Initialize face recognition model
        self.recognizer_net = cv2.dnn.readNetFromONNX(FACE_NET_MODEL_PATH)
        
        # Load known face embeddings
        self.known_embeddings = []
        self.known_names = []
        self._load_known_faces()
        
        # Initialize FAISS index
        if len(self.known_embeddings) > 0:
            self.known_embeddings = np.vstack(self.known_embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(self.known_embeddings.shape[1])
            self.index.add(self.known_embeddings)
        else:
            raise ValueError("No face embeddings found")

    def _load_known_faces(self):
        """Load and compute embeddings for known faces."""
        for person_name in os.listdir(FACE_FOLDER):
            person_folder = os.path.join(FACE_FOLDER, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self._process_known_face(os.path.join(person_folder, filename), person_name)

    def _process_known_face(self, img_path, person_name):
        """Process a single known face image."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image {img_path}")
            return

        self.yunet.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.yunet.detect(img)
        
        if faces is not None and len(faces) > 0:
            face = faces[0]
            landmarks = face[4:14].reshape((5, 2))
            aligned_face = align_face(img, landmarks)
            embedding = self._compute_face_embedding(aligned_face)
            self.known_embeddings.append(embedding)
            self.known_names.append(person_name)

    def _compute_face_embedding(self, aligned_face):
        """Compute embedding for a face image."""
        blob = cv2.dnn.blobFromImage(
            aligned_face, 
            scalefactor=1.0/127.5, 
            size=(112, 112),
            mean=(127.5, 127.5, 127.5), 
            swapRB=True, 
            crop=False
        )
        self.recognizer_net.setInput(blob)
        embedding = self.recognizer_net.forward()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in a frame."""
        small_frame = cv2.resize(frame, FACE_DETECTION_INPUT_SIZE)
        height, width, _ = small_frame.shape
        self.yunet.setInputSize((width, height))
        _, faces = self.yunet.detect(small_frame)
        
        detections = []
        if faces is not None:
            for face in faces:
                detection = self._process_detected_face(face, frame, small_frame)
                if detection:
                    detections.append(detection)
        
        return detections

    def _process_detected_face(self, face, frame, small_frame):
        """Process a single detected face."""
        bbox = face[:4]
        landmarks = face[4:14].reshape((5, 2))
        
        # Scale coordinates to original frame size
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        bbox_scaled = bbox * [scale_x, scale_y, scale_x, scale_y]
        bbox_scaled = bbox_scaled.astype(np.int32)
        
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
        
        # Align and compute embedding
        aligned_face = align_face(frame, landmarks)
        face_embedding = self._compute_face_embedding(aligned_face)
        
        # Find closest match
        D, I = self.index.search(face_embedding.reshape(1, -1), k=1)
        distance = D[0][0]
        idx = I[0][0]
        
        # Determine if face is recognized
        name = self.known_names[idx] if distance < FACE_RECOGNITION_THRESHOLD else 'unknown'
        recognized = distance < FACE_RECOGNITION_THRESHOLD
        
        return {
            'bbox': bbox_scaled,
            'name': name,
            'recognized': recognized
        }