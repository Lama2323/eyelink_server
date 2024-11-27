import cv2
import numpy as np
import os
import faiss
import shutil
from src.database.supabase_client import supabase

# Global variables
known_embeddings = []
known_names = []
index = None

# Reference points for face alignment
reference_points = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, landmarks):
    """Align face using landmarks"""
    src_pts = landmarks.astype(np.float32)
    dst_pts = reference_points
    tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    aligned_face = cv2.warpAffine(img, tform, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned_face

def sync_face_folder():
    """Sync local face folder with Supabase bucket"""
    local_face_dir = "face"
    
    if os.path.exists(local_face_dir):
        shutil.rmtree(local_face_dir)
    
    os.makedirs(local_face_dir)
    
    try:
        response = supabase.storage.from_('face').list()
        
        for folder in response:
            folder_name = folder['name']
            local_subfolder = os.path.join(local_face_dir, folder_name)
            os.makedirs(local_subfolder)
            
            files = supabase.storage.from_('face').list(folder_name)
            
            for file in files:
                if file['name'].lower().endswith('.jpg'):
                    file_path = f"{folder_name}/{file['name']}"
                    data = supabase.storage.from_('face').download(file_path)
                    
                    local_file_path = os.path.join(local_subfolder, file['name'])
                    with open(local_file_path, 'wb') as f:
                        f.write(data)
                    
        print("Face folder sync successful!")
        
    except Exception as e:
        print(f"Error syncing face folder: {str(e)}")
        if os.path.exists(local_face_dir):
            shutil.rmtree(local_face_dir)
        raise e

def load_face_recognition():
    """Load and initialize face recognition system"""
    global known_embeddings, known_names, index
    
    known_embeddings = []
    known_names = []
    face_folder = 'face'
    
    yunet = cv2.FaceDetectorYN.create(
        model="model/yunet.onnx",
        config="",
        input_size=(160, 160),
        score_threshold=0.6,
        nms_threshold=0.4,
        top_k=5000
    )
    
    recognizer_net = cv2.dnn.readNetFromONNX('model/mobilefacenet.onnx')
    
    d = 128
    index = faiss.IndexFlatL2(d)
    
    if os.path.exists(face_folder):
        # Process each person's folder
        for person_name in os.listdir(face_folder):
            person_folder = os.path.join(face_folder, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_folder, filename)
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Cannot read image {img_path}")
                            continue
                        
                        yunet.setInputSize((img.shape[1], img.shape[0]))
                        _, faces = yunet.detect(img)
                        
                        if faces is not None and len(faces) > 0:
                            face = faces[0]
                            landmarks = face[4:14].reshape((5, 2))
                            
                            aligned_face = align_face(img, landmarks)
                            blob = cv2.dnn.blobFromImage(aligned_face, 
                                                       scalefactor=1.0/127.5,
                                                       size=(112, 112),
                                                       mean=(127.5, 127.5, 127.5),
                                                       swapRB=True, 
                                                       crop=False)
                            
                            recognizer_net.setInput(blob)
                            face_embedding = recognizer_net.forward()
                            face_embedding = face_embedding / np.linalg.norm(face_embedding)
                            face_embedding = face_embedding.flatten()
                            
                            known_embeddings.append(face_embedding)
                            known_names.append(person_name)
    
    if len(known_embeddings) > 0:
        known_embeddings = np.vstack(known_embeddings).astype('float32')
        index.reset()
        index.add(known_embeddings)
        return True
    return False