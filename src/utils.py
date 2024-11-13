import cv2
import numpy as np
from .config import REFERENCE_POINTS

def align_face(img, landmarks):
    """Align face using landmarks."""
    src_pts = landmarks.astype(np.float32)
    dst_pts = REFERENCE_POINTS
    tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    aligned_face = cv2.warpAffine(img, tform, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned_face

def compute_iou(box1, box2):
    """Compute Intersection over Union between two bounding boxes."""
    x1, y1, x1w, y1h = box1
    x2, y2, x2w, y2h = box2
    x1_max = x1 + x1w
    y1_max = y1 + y1h
    x2_max = x2 + x2w
    y2_max = y2 + y2h
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = x1w * y1h
    box2_area = x2w * y2h
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def draw_detections(frame, tracked_faces):
    """Draw bounding boxes and labels for detected faces."""
    for tracked_face in tracked_faces.values():
        x, y, w, h = tracked_face.bbox
        color = (0, 255, 0) if tracked_face.recognized else (0, 0, 255)
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
        
        # Add background for name label
        label_size = cv2.getTextSize(tracked_face.name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(frame, (int(x), int(y - 30)), 
                     (int(x + label_size[0]), int(y)), color, -1)
        
        # Add name label
        cv2.putText(frame, tracked_face.name, (int(x), int(y - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return frame