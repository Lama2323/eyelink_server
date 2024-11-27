from src.app.gui import ModernFaceDetectionApp
import cv2

def main():
    """Main function to run the application"""
    app = ModernFaceDetectionApp()
    
    # Create OpenCV window
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    
    # Initialize program
    app.set_initial_camera_source()
    app.update_frame()
    
    # Start main loop
    app.root.mainloop()

if __name__ == "__main__":
    main()