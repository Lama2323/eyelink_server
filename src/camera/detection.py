import cv2
import numpy as np
from src.face.recognition import align_face
import src.face.recognition as face_recognition  # Import as module

def detect_faces(latest_frame, frame_lock, latest_result, result_lock, stop_event, yunet, recognizer_net):
    """Thread function to detect and recognize faces"""
    while not stop_event.is_set():
        with frame_lock:
            if latest_frame[0] is None:
                continue
            frame = latest_frame[0].copy()
        
        # Resize and detect faces
        small_frame = cv2.resize(frame, (160, 160))
        height, width, _ = small_frame.shape
        yunet.setInputSize((width, height))
        _, faces = yunet.detect(small_frame)
        
        detections = []
        if faces is not None:
            for face in faces:
                # Process face detection
                bbox = face[:4]
                landmarks = face[4:14].reshape((5, 2))
                
                # Scale coordinates
                scale_x = frame.shape[1] / width
                scale_y = frame.shape[0] / height
                bbox_scaled = bbox * [scale_x, scale_y, scale_x, scale_y]
                bbox_scaled = bbox_scaled.astype(np.int32)
                landmarks[:, 0] *= scale_x
                landmarks[:, 1] *= scale_y
                
                # Align and get face embedding
                aligned_face = align_face(frame, landmarks)
                blob = cv2.dnn.blobFromImage(aligned_face, 
                                           scalefactor=1.0/127.5,
                                           size=(112, 112),
                                           mean=(127.5, 127.5, 127.5),
                                           swapRB=True,
                                           crop=False)
                
                recognizer_net.setInput(blob)
                face_embedding = recognizer_net.forward()
                face_embedding = face_embedding / np.linalg.norm(face_embedding)
                face_embedding = face_embedding.flatten().astype('float32')
                
                # Check if we have known faces to compare against
                if len(face_recognition.known_embeddings) > 0:
                    D, I = face_recognition.index.search(face_embedding.reshape(1, -1), k=1)
                    distance = D[0][0]
                    idx = I[0][0]
                    threshold = 1.05
                    
                    if distance < threshold:
                        name = face_recognition.known_names[idx]
                        recognized = True
                    else:
                        name = 'unknown'
                        recognized = False
                else:
                    name = 'unknown'
                    recognized = False
                
                detections.append({'bbox': bbox_scaled, 'name': name, 'recognized': recognized})
        
        with result_lock:
            latest_result[0] = (frame, detections)