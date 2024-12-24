import cv2
from gui.app import ModernFaceDetectionApp

def main():
    app = ModernFaceDetectionApp()
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    app.set_initial_camera_source()
    app.update_frame()
    app.root.mainloop()

if __name__ == "__main__":
    main()