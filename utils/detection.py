import cv2
import numpy as np
import faiss
import os
from utils.alignment import align_face

class FaceRecognitionData:
    def __init__(self):
        self.known_embeddings = []
        self.known_names = []
        self.index = faiss.IndexFlatL2(128)

face_recognition_data = FaceRecognitionData()

def load_face_recognition():
    face_recognition_data.known_embeddings = []
    face_recognition_data.known_names = []
    face_folder = 'face'

    yunet = cv2.FaceDetectorYN.create(
        model="model/yunet.onnx",
        config="",
        input_size=(160, 160),
        score_threshold=0.65,
        nms_threshold=0.4,
        top_k=50
    )

    recognizer_net = cv2.dnn.readNetFromONNX('model/mobilefacenet.onnx')

    if os.path.exists(face_folder):
        for person_name in os.listdir(face_folder):
            person_folder = os.path.join(face_folder, person_name)
            if os.path.isdir(person_folder):
                person_embeddings = []
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
                                                       scalefactor=1.0 / 127.5,
                                                       size=(112, 112),
                                                       mean=(127.5, 127.5, 127.5),
                                                       swapRB=True,
                                                       crop=False)

                            recognizer_net.setInput(blob)
                            face_embedding = recognizer_net.forward()
                            face_embedding = face_embedding / np.linalg.norm(face_embedding)
                            face_embedding = face_embedding.flatten()

                            person_embeddings.append(face_embedding)

                if len(person_embeddings) > 0:
                    avg_embedding = np.mean(person_embeddings, axis=0)
                    face_recognition_data.known_embeddings.append(avg_embedding)
                    face_recognition_data.known_names.append(person_name)

    if len(face_recognition_data.known_embeddings) > 0:
        known_embeddings_np = np.vstack(face_recognition_data.known_embeddings).astype('float32')
        face_recognition_data.index.reset()
        face_recognition_data.index.add(known_embeddings_np)
        return True
    return False

def detect_faces(latest_frame, frame_lock, latest_result, result_lock, stop_event, yunet, recognizer_net):
    while not stop_event.is_set():
        with frame_lock:
            if latest_frame[0] is None:
                continue
            frame = latest_frame[0].copy()
        
        # Chuyển sang HSV để phân tích điều kiện ánh sáng
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        
        # Tính histogram của kênh Value để đánh giá điều kiện ánh sáng
        hist = cv2.calcHist([v], [0], None, [256], [0, 256])
        mean_brightness = np.mean(v)
        std_brightness = np.std(v)
        
        # Chỉ áp dụng equalization nếu:
        # - Độ sáng trung bình thấp (< 85) hoặc
        # - Độ tương phản kém (std < 30)
        if mean_brightness < 85 or std_brightness < 30:
            equalized_v = cv2.equalizeHist(v)
            hsv[:,:,2] = equalized_v
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        small_frame = cv2.resize(frame, (160, 160))
        height, width, _ = small_frame.shape
        yunet.setInputSize((width, height))
        _, faces = yunet.detect(small_frame)

        detections = []
        if faces is not None:
            for face in faces:
                bbox = face[:4]
                landmarks = face[4:14].reshape((5, 2))

                scale_x = frame.shape[1] / width
                scale_y = frame.shape[0] / height
                bbox_scaled = bbox * [scale_x, scale_y, scale_x, scale_y]
                bbox_scaled = bbox_scaled.astype(np.int32)
                landmarks[:, 0] *= scale_x
                landmarks[:, 1] *= scale_y

                aligned_face = align_face(frame, landmarks)
                blob = cv2.dnn.blobFromImage(aligned_face,
                                           scalefactor=1.0 / 127.5,
                                           size=(112, 112),
                                           mean=(127.5, 127.5, 127.5),
                                           swapRB=True,
                                           crop=False)

                recognizer_net.setInput(blob)
                face_embedding = recognizer_net.forward()
                face_embedding = face_embedding / np.linalg.norm(face_embedding)
                face_embedding = face_embedding.flatten().astype('float32')

                if len(face_recognition_data.known_embeddings) > 0:
                    k = 3
                    D, I = face_recognition_data.index.search(face_embedding.reshape(1, -1), k=k)

                    counts = {}
                    for idx in I[0]:
                        name = face_recognition_data.known_names[idx]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)

                    avg_distance = np.mean(D[0][np.where(np.array(I[0]) == face_recognition_data.known_names.index(name))])

                    threshold = 1.05
                    if avg_distance < threshold:
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

def draw_detections(frame, tracked_faces):
    for tracked_face in tracked_faces.values():
        x, y, w, h = tracked_face.bbox

        if tracked_face.recognized:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame,
                     (int(x), int(y)),
                     (int(x + w), int(y + h)),
                     color, 2)

        label_size = cv2.getTextSize(tracked_face.name,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(frame,
                     (int(x), int(y - 30)),
                     (int(x + label_size[0]), int(y)),
                     color, -1)

        cv2.putText(frame, tracked_face.name,
                   (int(x), int(y - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   (255, 255, 255), 2)

    return frame