import cv2
import numpy as np
import faiss
from config.settings import *

class FaceRecognitionSystem:
    def __init__(self):
        self.known_embeddings = []
        self.known_names = []
        self.index = None
        self.recognizer_net = cv2.dnn.readNetFromONNX(FACE_RECOGNITION_MODEL_PATH)
    
    def align_face(self, img, landmarks):
        src_pts = landmarks.astype(np.float32)
        dst_pts = np.array(REFERENCE_POINTS, dtype=np.float32)
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        aligned_face = cv2.warpAffine(img, tform, FACE_RECOGNITION_SIZE, flags=cv2.INTER_LINEAR)
        return aligned_face
    
    def get_face_embedding(self, aligned_face):
        blob = cv2.dnn.blobFromImage(
            aligned_face, 
            scalefactor=FACE_RECOGNITION_SCALE,
            size=FACE_RECOGNITION_SIZE,
            mean=FACE_RECOGNITION_MEAN,
            swapRB=True,
            crop=False
        )
        self.recognizer_net.setInput(blob)
        face_embedding = self.recognizer_net.forward()
        face_embedding = face_embedding / np.linalg.norm(face_embedding)
        return face_embedding.flatten()

    def initialize_index(self, embeddings):
        if len(embeddings) > 0:
            d = embeddings[0].shape[0]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(np.vstack(embeddings).astype('float32'))
            return True
        return False