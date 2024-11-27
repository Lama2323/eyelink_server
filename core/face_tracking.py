import time
from models.tracked_face import TrackedFace
from utils.geometry import compute_iou
from config.settings import *

class FaceTracker:
    def __init__(self):
        self.tracked_faces = {}
        self.face_id_counter = 0
    
    def update_tracks(self, detections, recognition_results):
        current_time = time.time()
        new_tracked_faces = {}
        
        for detection, recognition in zip(detections, recognition_results):
            bbox = detection['bbox']
            name = recognition['name']
            recognized = recognition['recognized']
            
            matched_face_id = self._match_detection(bbox)
            
            if matched_face_id is not None:
                tracked_face = self._update_existing_track(
                    matched_face_id, bbox, name, recognized, current_time
                )
                new_tracked_faces[matched_face_id] = tracked_face
            else:
                new_face_id = self._create_new_track(bbox, name, recognized, current_time)
                new_tracked_faces[new_face_id] = self.tracked_faces[new_face_id]
        
        self._cleanup_old_tracks(current_time)
        self.tracked_faces = new_tracked_faces
        
        return self.tracked_faces
    
    def _match_detection(self, bbox):
        matched_face_id = None
        max_iou = 0
        
        for face_id, tracked_face in self.tracked_faces.items():
            iou = compute_iou(bbox, tracked_face.bbox)
            if iou > TRACKING_IOU_THRESHOLD and iou > max_iou:
                max_iou = iou
                matched_face_id = face_id
        
        return matched_face_id
    
    def _update_existing_track(self, face_id, bbox, name, recognized, current_time):
        tracked_face = self.tracked_faces[face_id]
        tracked_face.update(bbox, name, recognized, current_time)
        return tracked_face
    
    def _create_new_track(self, bbox, name, recognized, current_time):
        self.face_id_counter += 1
        self.tracked_faces[self.face_id_counter] = TrackedFace(
            self.face_id_counter, bbox, name, recognized, current_time
        )
        return self.face_id_counter
    
    def _cleanup_old_tracks(self, current_time):
        self.tracked_faces = {
            face_id: face for face_id, face in self.tracked_faces.items()
            if current_time - face.last_update_time <= TRACKING_TIMEOUT
        }