class TrackedFace:
    """Class to track face detection states"""
    def __init__(self, face_id, bbox, name, recognized, timestamp):
        self.face_id = face_id
        self.bbox = bbox
        self.name = name
        self.recognized = recognized
        self.state_duration = 0
        self.last_update_time = timestamp
        self.current_state_start_time = timestamp
        self.unknown_duration = 0