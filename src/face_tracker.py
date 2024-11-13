import time
from .utils import compute_iou
from .config import FACE_TRACK_TIMEOUT, UNKNOWN_FACE_TIMEOUT

class TrackedFace:
    def __init__(self, face_id, bbox, name, recognized, timestamp):
        self.face_id = face_id
        self.bbox = bbox
        self.name = name
        self.recognized = recognized
        self.state_duration = 0
        self.last_update_time = timestamp
        self.current_state_start_time = timestamp
        self.unknown_duration = 0

class FaceTracker:
    def __init__(self):
        self.tracked_faces = {}
        self.face_id_counter = 0
    
    def update(self, detections):
        """Update tracked faces with new detections."""
        current_time = time.time()
        new_tracked_faces = {}
        
        # Update existing tracks with new detections
        for detection in detections:
            bbox = detection['bbox']
            name = detection['name']
            recognized = detection['recognized']
            
            matched_face_id = self._find_matching_face(bbox)
            
            if matched_face_id is not None:
                self._update_existing_face(matched_face_id, bbox, name, recognized, current_time, new_tracked_faces)
            else:
                self._create_new_face(bbox, name, recognized, current_time, new_tracked_faces)
        
        # Remove old tracks
        self.tracked_faces = {
            face_id: face 
            for face_id, face in new_tracked_faces.items()
            if current_time - face.last_update_time <= FACE_TRACK_TIMEOUT
        }
        
        return self.tracked_faces
    
    def _find_matching_face(self, bbox):
        """Find matching face using IOU."""
        matched_face_id = None
        max_iou = 0
        
        for face_id, tracked_face in self.tracked_faces.items():
            iou = compute_iou(bbox, tracked_face.bbox)
            if iou > 0.5 and iou > max_iou:
                max_iou = iou
                matched_face_id = face_id
        
        return matched_face_id
    
    def _update_existing_face(self, face_id, bbox, name, recognized, current_time, new_tracked_faces):
        """Update an existing tracked face."""
        tracked_face = self.tracked_faces[face_id]
        tracked_face.bbox = bbox
        time_since_last_update = current_time - tracked_face.last_update_time
        
        if tracked_face.recognized == recognized and tracked_face.name == name:
            tracked_face.state_duration += time_since_last_update
            tracked_face.unknown_duration = 0
        else:
            if tracked_face.recognized and not recognized:
                tracked_face.unknown_duration += time_since_last_update
                if tracked_face.unknown_duration >= UNKNOWN_FACE_TIMEOUT:
                    self._update_face_state(tracked_face, name, recognized, current_time)
            else:
                self._update_face_state(tracked_face, name, recognized, current_time)
        
        tracked_face.last_update_time = current_time
        new_tracked_faces[face_id] = tracked_face
    
    def _create_new_face(self, bbox, name, recognized, current_time, new_tracked_faces):
        """Create a new tracked face."""
        self.face_id_counter += 1
        tracked_face = TrackedFace(self.face_id_counter, bbox, name, recognized, current_time)
        new_tracked_faces[self.face_id_counter] = tracked_face
    
    def _update_face_state(self, tracked_face, name, recognized, current_time):
        """Update the state of a tracked face."""
        tracked_face.name = name
        tracked_face.recognized = recognized
        tracked_face.state_duration = 0
        tracked_face.current_state_start_time = current_time
        tracked_face.unknown_duration = 0
    
    def get_statistics(self):
        """Get current tracking statistics."""
        num_unknown = 0
        known_names_set = set()
        
        for tracked_face in self.tracked_faces.values():
            if tracked_face.recognized:
                known_names_set.add(tracked_face.name)
            else:
                num_unknown += 1
        
        return {
            'unknown_count': num_unknown,
            'known_faces': list(known_names_set)
        }