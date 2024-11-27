import cv2
import customtkinter as ctk
from ui.app import ModernFaceDetectionApp

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)

    app = ModernFaceDetectionApp()
    app.run()

if __name__ == "__main__":
    main()