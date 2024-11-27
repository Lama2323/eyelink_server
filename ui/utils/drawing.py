import cv2

def draw_detections(frame, tracked_faces):
    for tracked_face in tracked_faces.values():
        x, y, w, h = tracked_face.bbox
        
        color = (0, 255, 0) if tracked_face.recognized else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, 
                     (int(x), int(y)), 
                     (int(x + w), int(y + h)), 
                     color, 2)
        
        # Add background for label
        label_size = cv2.getTextSize(tracked_face.name, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(frame, 
                     (int(x), int(y - 30)), 
                     (int(x + label_size[0]), int(y)), 
                     color, -1)
        
        # Add name label
        cv2.putText(frame, tracked_face.name,
                   (int(x), int(y - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   (255, 255, 255), 2)
    
    return frame