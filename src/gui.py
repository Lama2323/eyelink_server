import cv2
import customtkinter as ctk
from .config import *

class ModernFaceDetectionApp:
    def __init__(self, video_handler):
        self.video_handler = video_handler
        self.streams = []
        self.current_stream_index = 0
        
        # Configure appearance
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(COLOR_THEME)
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title(GUI_WINDOW_TITLE)
        self.root.geometry(GUI_WINDOW_SIZE)
        
        self._init_gui_components()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _init_gui_components(self):
        """Initialize GUI components."""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text=GUI_TITLE,
            font=("Helvetica", 24)
        )
        self.title_label.pack(pady=20)
        
        # Stats Frame
        self._init_stats_frame()
        
        # Control Frame
        self._init_control_frame()
        
        # Status Frame
        self._init_status_frame()
    
    def _init_stats_frame(self):
        """Initialize statistics frame."""
        self.stats_frame = ctk.CTkFrame(self.main_frame)
        self.stats_frame.pack(fill="x", padx=10, pady=10)
        
        self.stranger_label = ctk.CTkLabel(
            self.stats_frame,
            text="Stranger: 0",
            font=("Helvetica", 16)
        )
        self.stranger_label.pack(pady=10)
        
        self.familiar_label = ctk.CTkLabel(
            self.stats_frame,
            text="Familiar face: ",
            font=("Helvetica", 16)
        )
        self.familiar_label.pack(pady=10)
        
        self.fps_label = ctk.CTkLabel(
            self.stats_frame,
            text="FPS: ...",
            font=("Helvetica", 16)
        )
        self.fps_label.pack(pady=10)
    
    def _init_control_frame(self):
        """Initialize control frame."""
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill="x", padx=10, pady=20)
        
        self.add_camera_button = ctk.CTkButton(
            self.control_frame,
            text="Add Camera",
            command=self.add_camera,
            font=("Helvetica", 14)
        )
        self.add_camera_button.pack(pady=10, padx=20, fill="x")
        
        # Navigation Frame
        self.nav_frame = ctk.CTkFrame(self.control_frame)
        self.nav_frame.pack(fill="x", pady=10)
        
        self.previous_button = ctk.CTkButton(
            self.nav_frame,
            text="Previous",
            command=self.previous_camera,
            font=("Helvetica", 14)
        )
        self.previous_button.pack(side="left", padx=5, expand=True)
        
        self.next_button = ctk.CTkButton(
            self.nav_frame,
            text="Next",
            command=self.next_camera,
            font=("Helvetica", 14)
        )
        self.next_button.pack(side="right", padx=5, expand=True)
    
    def _init_status_frame(self):
        """Initialize status frame."""
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=10)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="System Status: Ready",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=10)
    
    def set_camera_source(self):
        """Show dialog to get camera source from user."""
        dialog = ctk.CTkInputDialog(
            text="Enter 0 for default camera or HTTP URL for stream:",
            title="Add Camera Source"
        )
        camera_source = dialog.get_input()
        
        if camera_source is None or camera_source.strip() == "":
            return None
        
        if camera_source == "0":
            return 0
        else:
            if not camera_source.startswith('http'):
                camera_source = f'http://{camera_source}:8080/video'
            return camera_source
    
    def set_initial_camera_source(self):
        """Set initial camera source on startup."""
        camera_source = self.set_camera_source()
        if camera_source is None:
            print("No camera source provided. Exiting.")
            exit()
        self.streams.append(camera_source)
    
    def add_camera(self):
        """Add a new camera source."""
        new_source = self.set_camera_source()
        if new_source is not None:
            self.streams.append(new_source)
            self.status_label.configure(text=f"Added new stream source: {new_source}")
    
    def previous_camera(self):
        """Switch to previous camera."""
        if len(self.streams) > 1:
            self.video_handler.stop_stream()
            self.current_stream_index = (self.current_stream_index - 1) % len(self.streams)
            self.video_handler.start_stream(self.streams[self.current_stream_index])
            self.status_label.configure(text=f"Switched to camera {self.current_stream_index + 1}")
    
    def next_camera(self):
        """Switch to next camera."""
        if len(self.streams) > 1:
            self.video_handler.stop_stream()
            self.current_stream_index = (self.current_stream_index + 1) % len(self.streams)
            self.video_handler.start_stream(self.streams[self.current_stream_index])
            self.status_label.configure(text=f"Switched to camera {self.current_stream_index + 1}")
    
    def update_stats(self, stats, fps):
        """Update statistics display."""
        if stats:
            self.stranger_label.configure(text=f"Stranger: {stats['unknown_count']}")
            self.familiar_label.configure(text=f"Familiar face: {', '.join(stats['known_faces'])}")
        self.fps_label.configure(text=f"FPS: {fps:.2f}")
    
    def update_frame(self):
        """Update video frame and statistics."""
        frame_with_detections, fps, stats = self.video_handler.get_processed_frame()
        
        if frame_with_detections is not None:
            cv2.imshow(OPENCV_WINDOW_NAME, frame_with_detections)
            cv2.waitKey(1)
            
            if stats:
                self.update_stats(stats, fps)
        
        self.root.after(10, self.update_frame)
    
    def on_closing(self):
        """Handle window closing."""
        self.video_handler.stop_stream()
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        # Create OpenCV window
        cv2.namedWindow(OPENCV_WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        # Initialize camera
        self.set_initial_camera_source()
        self.video_handler.start_stream(self.streams[self.current_stream_index])
        
        # Start update loop
        self.update_frame()
        
        # Start main loop
        self.root.mainloop()