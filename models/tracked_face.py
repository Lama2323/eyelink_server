from config.settings import *

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
    
    def update(self, bbox, name, recognized, current_time):
        time_since_last_update = current_time - self.last_update_time
        
        if self.recognized == recognized and self.name == name:
            self.state_duration += time_since_last_update
        else:
            if self.recognized and not recognized:
                self.unknown_duration += time_since_last_update
                if self.unknown_duration >= UNKNOWN_DURATION_THRESHOLD:
                    self._change_state(name, recognized, current_time)
            else:
                self._change_state(name, recognized, current_time)
        
        self.bbox = bbox
        self.last_update_time = current_time
    
    def _change_state(self, name, recognized, current_time):
        self.name = name
        self.recognized = recognized
        self.state_duration = 0
        self.current_state_start_time = current_time
        self.unknown_duration = 0