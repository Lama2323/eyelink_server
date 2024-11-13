import cv2
from src.face_detector import FaceDetector
from src.face_tracker import FaceTracker
from src.video_handler import VideoHandler
from src.gui import ModernFaceDetectionApp

def main():
    # Enable OpenCL
    cv2.ocl.setUseOpenCL(True)
    
    # Initialize components
    face_detector = FaceDetector()
    face_tracker = FaceTracker()
    video_handler = VideoHandler(face_detector, face_tracker)
    
    # Create and run GUI application
    app = ModernFaceDetectionApp(video_handler)
    app.run()

if __name__ == "__main__":
    main()