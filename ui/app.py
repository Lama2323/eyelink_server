import cv2
import time
import customtkinter as ctk
from .components.stats_frame import StatsFrame
from .components.control_frame import ControlFrame
from .components.status_frame import StatusFrame
from .utils.drawing import draw_detections
from services.camera_service import CameraStream
from services.storage_service import StorageService
from services.logger_service import FaceDetectionLogger
from core.face_recognition import FaceRecognitionSystem


class ModernFaceDetectionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Face Detection System")
        self.root.geometry("400x700")
        
        self._camera_sources = []
        self.camera_streams = []
        self.current_camera_index = 0
        self.last_stats_time = time.time()
        
        self.storage_service = StorageService()
        self.logger = FaceDetectionLogger()
        self.face_recognition = FaceRecognitionSystem()
        
        self._setup_ui()
        self._init_system()
    
    def _setup_ui(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Face Detection Control Panel",
            font=("Helvetica", 24)
        )
        self.title_label.pack(pady=20)
        
        # Components
        self.stats_frame = StatsFrame(self.main_frame)
        self.stats_frame.pack(fill="x", padx=10, pady=10)
        
        self.control_frame = ControlFrame(
            self.main_frame,
            {
                'refresh': self.refresh_faces,
                'add_camera': self.add_camera,
                'remove_camera': self.remove_camera,
                'previous': self.previous_camera,
                'next': self.next_camera
            }
        )
        self.control_frame.pack(fill="x", padx=10, pady=20)
        
        self.status_frame = StatusFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=10)
        
        # Set window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _init_system(self):
        """Initialize the face recognition system"""
        self.status_frame.update_status("Syncing faces with Supabase...")
        self.root.update()
        
        try:
            self.storage_service.sync_faces()
            self.status_frame.update_status("Loading face recognition system...")
            self.root.update()
            
            success = self.face_recognition.initialize_index([])  # Initialize with loaded embeddings
            
            status = ("Face recognition system initialized with known faces" 
                     if success else "System running in detection-only mode (no known faces)")
            self.status_frame.update_status(status)
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            self.status_frame.update_status("Running in detection-only mode (initialization failed)")
    
    def set_camera_source(self):
        """Get camera source from user input"""
        dialog = ctk.CTkInputDialog(
            text="Enter 0 for default camera or HTTP URL for stream:",
            title="Add Camera Source"
        )
        camera_source = dialog.get_input()
        if camera_source is None or camera_source.strip() == "":
            return None
            
        source = 0 if camera_source == "0" else camera_source
        if not str(source).startswith('http') and str(source) != "0":
            source = 'http://' + camera_source + ':8080/video'
        
        return source
    
    def _init_camera(self):
        """Initialize the first camera"""
        camera_source = self.set_camera_source()
        if camera_source is None:
            print("No camera source provided. Exiting.")
            self.root.destroy()
            return
            
        self._camera_sources.append(camera_source)
        camera_id = len(self.camera_streams) + 1
        first_camera = CameraStream(camera_source, camera_id)
        
        if first_camera.start():
            self.camera_streams.append(first_camera)
            self.status_frame.update_status(f"Camera initialized: {camera_source}")
        else:
            self.status_frame.update_status(f"Failed to initialize camera: {camera_source}")
            self.root.destroy()
    
    def add_camera(self, preset_source=None):
        """Add a new camera stream"""
        if preset_source is not None:
            new_source = preset_source
        else:
            new_source = self.set_camera_source()
            
        if new_source is not None:
            camera_id = len(self.camera_streams) + 1
            new_camera = CameraStream(new_source, camera_id)
            if new_camera.start():
                self.camera_streams.append(new_camera)
                if preset_source is None:
                    self._camera_sources.append(new_source)
                self.status_frame.update_status(f"Added camera: {new_source}")
            else:
                self.status_frame.update_status(f"Failed to add camera: {new_source}")
    
    def remove_camera(self):
        """Remove the last camera stream"""
        if len(self.camera_streams) > 0:
            camera_stream = self.camera_streams.pop()
            camera_stream.stop()
            if self._camera_sources:
                self._camera_sources.pop()
            self.status_frame.update_status(f"Removed camera {camera_stream.camera_id}")
            
            if self.current_camera_index >= len(self.camera_streams):
                self.current_camera_index = max(0, len(self.camera_streams) - 1)
        else:
            self.status_frame.update_status("No cameras to remove")
    
    def previous_camera(self):
        """Switch to previous camera"""
        if len(self.camera_streams) > 1:
            self.current_camera_index = (self.current_camera_index - 1) % len(self.camera_streams)
            self.status_frame.update_status(f"Switched to camera {self.current_camera_index + 1}")
    
    def next_camera(self):
        """Switch to next camera"""
        if len(self.camera_streams) > 1:
            self.current_camera_index = (self.current_camera_index + 1) % len(self.camera_streams)
            self.status_frame.update_status(f"Switched to camera {self.current_camera_index + 1}")
    
    def refresh_faces(self):
        """Refresh face recognition system"""
        stored_sources = self._camera_sources.copy()
        
        # Stop all cameras
        for camera_stream in self.camera_streams:
            camera_stream.stop()
        
        self.camera_streams = []
        self.current_camera_index = 0
        
        # Reinitialize system
        self._init_system()
        
        # Restore cameras
        self._camera_sources = []
        for source in stored_sources:
            self.add_camera(source)
        
        self.status_frame.update_status("Face recognition system refreshed")
    
    def update_frame(self):
        """Update frame display and face tracking"""
        current_time = time.time()
        total_tracked_faces = {}
        num_unknown_total = 0
        known_names_set = set()
        frame_to_display = None
        
        for idx, camera_stream in enumerate(self.camera_streams):
            with camera_stream.result_lock:
                if camera_stream.latest_result[0] is not None:
                    frame, tracked_faces = camera_stream.latest_result[0]
                    camera_stream.latest_result[0] = None
                    
                    # Update statistics
                    for tracked_face in tracked_faces.values():
                        if tracked_face.recognized:
                            known_names_set.add(tracked_face.name)
                        else:
                            num_unknown_total += 1
                    
                    # Prepare frame for display
                    if idx == self.current_camera_index:
                        frame_to_display = draw_detections(frame.copy(), tracked_faces)
        
        # Display frame
        if frame_to_display is not None:
            cv2.imshow('Face Detection', frame_to_display)
            cv2.waitKey(1)
        
        # Update stats
        if current_time - self.last_stats_time >= 1:
            # Update display
            average_fps = 0
            if self.camera_streams:
                average_fps = sum(stream.fps for stream in self.camera_streams) / len(self.camera_streams)
            self.stats_frame.update_stats(num_unknown_total, list(known_names_set), average_fps)
            
            # Update log if needed
            if self.logger.should_update(current_time, num_unknown_total, known_names_set):
                self.logger.update_log(num_unknown_total, list(known_names_set), current_time)
            
            self.last_stats_time = current_time
        
        # Schedule next update
        self.root.after(10, self.update_frame)
    
    def on_closing(self):
        """Handle application closing"""
        for camera_stream in self.camera_streams:
            camera_stream.stop()
        cv2.destroyAllWindows()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self._init_camera()
        self.update_frame()
        self.root.mainloop()