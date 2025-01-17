import time

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
        self.confidence_count = 1 
        self.missing_count = 0  

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou