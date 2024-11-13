import cv2
import numpy as np

# Appearance settings
APPEARANCE_MODE = "dark"
COLOR_THEME = "blue"

# Model paths
YUNET_MODEL_PATH = "model/yunet.onnx"
FACE_NET_MODEL_PATH = "model/mobilefacenet.onnx"

# Face detection parameters
FACE_DETECTION_INPUT_SIZE = (160, 160)
FACE_DETECTION_SCORE_THRESHOLD = 0.6
FACE_DETECTION_NMS_THRESHOLD = 0.4
FACE_DETECTION_TOP_K = 5000

# Face recognition parameters
FACE_RECOGNITION_THRESHOLD = 1.05

# Reference points for face alignment
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Video processing
FACE_FOLDER = 'face'
OPENCV_WINDOW_NAME = 'Face Detection'

# GUI settings
GUI_WINDOW_TITLE = "Face Detection System"
GUI_WINDOW_SIZE = "400x600"
GUI_TITLE = "Face Detection Control Panel"

# Time constants
STATS_UPDATE_INTERVAL = 1.0  # seconds
FACE_TRACK_TIMEOUT = 5.0    # seconds
UNKNOWN_FACE_TIMEOUT = 3.0  # seconds

# Colors
COLOR_RECOGNIZED = (0, 255, 0)  # Green
COLOR_UNKNOWN = (0, 0, 255)    # Red
COLOR_WHITE = (255, 255, 255)  # White