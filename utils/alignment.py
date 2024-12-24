import cv2
import numpy as np

reference_points = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, landmarks):
    src_pts = landmarks.astype(np.float32)
    dst_pts = reference_points
    tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    aligned_face = cv2.warpAffine(img, tform, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned_face