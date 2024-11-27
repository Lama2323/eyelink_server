# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_API_KEY')

# Model paths
YUNET_MODEL_PATH = "model/yunet.onnx"
FACE_RECOGNITION_MODEL_PATH = "model/mobilefacenet.onnx"

# Face detection settings
FACE_DETECTION_SIZE = (320, 320)
FACE_DETECTION_SCORE_THRESHOLD = 0.7
FACE_DETECTION_NMS_THRESHOLD = 0.3
FACE_DETECTION_TOP_K = 50

# Face recognition settings
FACE_RECOGNITION_SIZE = (112, 112)
FACE_RECOGNITION_MEAN = (127.5, 127.5, 127.5)
FACE_RECOGNITION_SCALE = 1.0/127.5

# Reference points for face alignment
REFERENCE_POINTS = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
]

# Tracking settings
TRACKING_IOU_THRESHOLD = 0.3
TRACKING_TIMEOUT = 5  # seconds
UNKNOWN_DURATION_THRESHOLD = 5  # seconds

# Logging settings
MIN_LOG_UPDATE_INTERVAL = 5  # seconds