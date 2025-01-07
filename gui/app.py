# app.py
import cv2
import threading
import time
import customtkinter as ctk
import re
from utils.detection import load_face_recognition, detect_faces, draw_detections, face_recognition_data
from utils.database import sync_face_folder, supabase
from utils.logging import FaceDetectionLogger
from utils.tracking import TrackedFace, compute_iou

class CameraStream:
    def __init__(self, stream_source, camera_id):
        self.stream_source = stream_source
        self.camera_id = camera_id
        self.stream = None
        self.stop_event = threading.Event()
        self.thread_read = None
        self.thread_detect = None
        self.latest_frame = [None]
        self.frame_lock = threading.Lock()
        self.latest_result = [None]
        self.result_lock = threading.Lock()
        self.tracked_faces = {}
        self.face_id_counter = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.init_complete = threading.Event()

        self.yunet = cv2.FaceDetectorYN.create(
            model="model/yunet.onnx",
            config="",
            input_size=(160, 160),
            score_threshold=0.6,
            nms_threshold=0.4,
            top_k=50
        )
        self.recognizer_net = cv2.dnn.readNetFromONNX('model/mobilefacenet.onnx')

    def start(self):
        self.stream = cv2.VideoCapture(self.stream_source)
        if not self.stream.isOpened():
            print(f"Cannot open stream {self.stream_source}")
            return False

        self.stop_event.clear()
        self.thread_read = threading.Thread(
            target=self.read_frames,
            args=(self.stream, self.latest_frame, self.frame_lock, self.stop_event)
        )
        self.thread_detect = threading.Thread(
            target=detect_faces,
            args=(self.latest_frame, self.frame_lock, self.latest_result,
                  self.result_lock, self.stop_event, self.yunet, self.recognizer_net)
        )

        self.thread_read.start()
        self.init_complete.set()
        self.thread_detect.start()
        return True

    def read_frames(self, stream, latest_frame, frame_lock, stop_event):
        while not stop_event.is_set():
            ret, frame = stream.read()
            if not ret:
                break
            with frame_lock:
                latest_frame[0] = frame.copy()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        if self.thread_read:
            self.thread_read.join()
        if self.thread_detect:
            self.thread_detect.join()
        if self.stream:
            self.stream.release()

        self.stop_event = threading.Event()
        self.thread_read = None
        self.thread_detect = None
        self.stream = None

class ModernFaceDetectionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Face Detection System")
        self.root.geometry("400x600")

        self._camera_sources = []
        self.logger = FaceDetectionLogger()
        self.camera_streams = []
        self.last_stats_time = time.time()
        self.current_camera_index = 0
        self.logged_in = False
        self.password_visible = False
        self.previous_stranger_count = -1
        self.previous_known_faces = frozenset()
        self.has_initial_camera = False

        self.login_frame = ctk.CTkFrame(self.root)
        self.login_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.login_title_label = ctk.CTkLabel(self.login_frame, text="EyeLink", font=("Helvetica", 24))
        self.login_title_label.pack(pady=(0, 20))

        self.email_label = ctk.CTkLabel(self.login_frame, text="Email", font=("Helvetica", 14))
        self.email_label.pack(pady=(0, 5), anchor="w")
        self.email_entry = ctk.CTkEntry(self.login_frame, font=("Helvetica", 14), height=40)
        self.email_entry.pack(pady=(0, 10), fill="x")

        self.password_label = ctk.CTkLabel(self.login_frame, text="Password", font=("Helvetica", 14))
        self.password_label.pack(pady=(0, 5), anchor="w")
        self.password_entry_frame = ctk.CTkFrame(self.login_frame)
        self.password_entry_frame.pack(pady=(0, 20), fill="x")
        self.password_entry = ctk.CTkEntry(self.password_entry_frame, show="*", font=("Helvetica", 14), height=40)
        self.password_entry.pack(side="left", fill="x", expand=True)
        self.password_toggle_button = ctk.CTkButton(self.password_entry_frame,
                                                   text="Hide",
                                                   width=40,
                                                   height=40,
                                                   command=self.toggle_password_visibility)
        self.password_toggle_button.pack(side="right")

        self.login_button = ctk.CTkButton(self.login_frame, text="Login", command=self.login, font=("Helvetica", 14))
        self.login_button.pack(pady=(0, 10))

        self.login_status_label = ctk.CTkLabel(self.login_frame, text="", fg_color="transparent", font=("Helvetica", 14))
        self.login_status_label.pack(pady=(0, 10))

        self.main_frame = ctk.CTkFrame(self.root)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="EyeLink Control Panel",
            font=("Helvetica", 24)
        )
        self.title_label.pack(pady=20)

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

        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill="x", padx=10, pady=20)

        self.refresh_button = ctk.CTkButton(
            self.control_frame,
            text="Refresh Faces",
            command=self.refresh_faces,
            font=("Helvetica", 14)
        )
        self.refresh_button.pack(pady=10, padx=20, fill="x")

        self.add_camera_button = ctk.CTkButton(
            self.control_frame,
            text="Add Camera",
            command=self.add_camera,
            font=("Helvetica", 14)
        )
        self.add_camera_button.pack(pady=10, padx=20, fill="x")

        self.remove_camera_button = ctk.CTkButton(
            self.control_frame,
            text="Remove Camera",
            command=self.remove_camera,
            font=("Helvetica", 14)
        )
        self.remove_camera_button.pack(pady=10, padx=20, fill="x")

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

        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="System Status: Awaiting login",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=10)

        self.logout_button = ctk.CTkButton(
            self.main_frame,
            text="Logout",
            command=self.logout,
            font=("Helvetica", 14)
        )
        self.logout_button.pack(pady=10, padx=20)
        self.logout_button.pack_forget() # Hide initially

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def is_valid_email(self, email):
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email)

    def toggle_password_visibility(self):
        self.password_visible = not self.password_visible
        if self.password_visible:
            self.password_entry.configure(show="")
            self.password_toggle_button.configure(text="Show")
        else:
            self.password_entry.configure(show="*")
            self.password_toggle_button.configure(text="Hide")

    def login(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get()

        if not self.is_valid_email(email):
            self.login_status_label.configure(text="Invalid email format", text_color="red")
            return

        self.login_status_label.configure(text="Logging in...", text_color="black")
        try:
            user = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if user:
                self.login_status_label.configure(text="Login successful!", text_color="green")
                self.logged_in = True
                self.login_frame.pack_forget()
                self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
                self.logout_button.pack(pady=10, padx=20)
                self.init_face_recognition()
                self.set_initial_camera_source()
                self.update_frame()
            else:
                self.login_status_label.configure(text="Login failed.", text_color="red")
        except Exception as e:
            self.login_status_label.configure(text=f"Login failed: {e}", text_color="red")

    def logout(self):
        try:
            supabase.auth.sign_out()
            self.logged_in = False

            if hasattr(self, 'camera_streams') and self.camera_streams:
                for camera_stream in self.camera_streams:
                    if camera_stream:
                        camera_stream.stop()

            self.camera_streams = []
            self._camera_sources = []
            self.current_camera_index = 0
            self.has_initial_camera = False

            self.main_frame.pack_forget()
            self.logout_button.pack_forget()
            self.login_frame.pack(fill="both", expand=True, padx=20, pady=20)
            self.status_label.configure(text="System Status: Awaiting login")
            self.login_status_label.configure(text="", fg_color="transparent")
            print("Logged out successfully")

        except Exception as e:
            print(f"Error during logout: {e}")

    def disable_buttons(self):
        self.refresh_button.configure(state="disabled")
        self.add_camera_button.configure(state="disabled")
        self.remove_camera_button.configure(state="disabled")

    def enable_buttons(self):
        self.refresh_button.configure(state="normal")
        self.add_camera_button.configure(state="normal")
        self.remove_camera_button.configure(state="normal")

    def init_face_recognition(self):
        self.disable_buttons()
        self.status_label.configure(text="Syncing faces with Supabase...")
        self.root.update()

        try:
            sync_face_folder()
            self.status_label.configure(text="Loading face recognition system...")
            self.root.update()

            load_face_recognition()

            if len(face_recognition_data.known_embeddings) > 0:
                self.status_label.configure(text="Face recognition system initialized with known faces")
            else:
                self.status_label.configure(text="System running in detection-only mode (no known faces)")

        except Exception as e:
            print(f"Error during Supabase sync: {str(e)}")
            self.status_label.configure(text="Running in detection-only mode (Supabase sync failed)")
        finally:
            self.enable_buttons()

    def refresh_faces(self):
        self.disable_buttons()
        stored_sources = self._camera_sources.copy()

        for camera_stream in self.camera_streams:
            camera_stream.stop()

        self.camera_streams = []
        self.current_camera_index = 0

        self.init_face_recognition()

        self._camera_sources = []
        for source in stored_sources:
            self.add_camera(source)

        self.status_label.configure(text="Face recognition system refreshed and cameras reloaded")
        self.enable_buttons()

    def set_camera_source(self):
        dialog = ctk.CTkInputDialog(
            text="Enter 0 for default camera, 1 for secondary camera, or HTTP URL for stream:",
            title="Add Camera Source"
        )
        camera_source = dialog.get_input()
        if camera_source is None or camera_source.strip() == "":
            return None

        if camera_source == "0":
            source = 0
        elif camera_source == "1":
            source = 1
        elif not str(camera_source).startswith('http'):
            source = 'http://' + camera_source + ':8080/video'
        else:
            source = camera_source

        return source
    
    def set_initial_camera_source(self):
        camera_source = self.set_camera_source()
        if camera_source is None:
            self.status_label.configure(text="No camera source provided. System running without cameras.")
            self.has_initial_camera = False
            return

        self._camera_sources.append(camera_source)
        camera_id = len(self.camera_streams) + 1
        first_camera = CameraStream(camera_source, camera_id)
        if first_camera.start():
            self.camera_streams.append(first_camera)
            self.has_initial_camera = True
        else:
            print(f"Failed to start stream {camera_source}")
            self.status_label.configure(text="Failed to start camera. System running without cameras.")
            self.has_initial_camera = False

    def add_camera(self, preset_source=None):
        if preset_source is not None:
            new_source = preset_source
        else:
            new_source = self.set_camera_source()

        if new_source is not None:
            # Kiểm tra xem source đã tồn tại chưa
            if new_source in self._camera_sources:
                self.status_label.configure(text=f"Camera already exists.")
                return

            camera_id = len(self.camera_streams) + 1
            new_camera = CameraStream(new_source, camera_id)
            if new_camera.start():
                self.camera_streams.append(new_camera)
                if preset_source is None:
                    self._camera_sources.append(new_source)
                self.status_label.configure(text=f"Added stream source: {new_source}")
            else:
                self.status_label.configure(text=f"Failed to start stream source: {new_source}")

    def remove_camera(self):
        if len(self.camera_streams) > 0:
            camera_stream = self.camera_streams.pop()
            camera_stream.stop()
            if self._camera_sources:
                self._camera_sources.pop()
            self.status_label.configure(text=f"Removed camera {camera_stream.camera_id}")
            if self.current_camera_index >= len(self.camera_streams):
                self.current_camera_index = max(0, len(self.camera_streams) - 1)
        else:
            self.status_label.configure(text="No cameras to remove")

    def previous_camera(self):
        if len(self.camera_streams) > 1:
            self.current_camera_index = (self.current_camera_index - 1) % len(self.camera_streams)
            self.status_label.configure(text=f"Switched to camera {self.current_camera_index + 1}")

    def next_camera(self):
        if len(self.camera_streams) > 1:
            self.current_camera_index = (self.current_camera_index + 1) % len(self.camera_streams)
            self.status_label.configure(text=f"Switched to camera {self.current_camera_index + 1}")

    def update_stats(self, num_strangers, known_names):
        if num_strangers != self.previous_stranger_count:
            self.stranger_label.configure(text=f"Stranger: {num_strangers}")
            self.previous_stranger_count = num_strangers

        if frozenset(known_names) != self.previous_known_faces:
            self.familiar_label.configure(text=f"Familiar face: {', '.join(known_names)}")
            self.previous_known_faces = frozenset(known_names)

    def on_closing(self):
        if self.camera_streams:
            for camera_stream in self.camera_streams:
                camera_stream.stop()
        cv2.destroyAllWindows()
        self.root.destroy()

    def update_frame(self):
        if not self.logged_in:
            self.root.after(10, self.update_frame)
            return

        if not self.has_initial_camera and len(self.camera_streams) == 0:
            self.root.after(10, self.update_frame)
            return

        current_time = time.time()
        total_tracked_faces = {}
        num_unknown_total = 0
        known_names_set = set()
        frame_to_display = None
        
        MIN_CONFIDENCE_FRAMES = 2 

        MAX_MISSING_FRAMES = 3    

        for idx, camera_stream in enumerate(self.camera_streams):
            with camera_stream.result_lock:
                if camera_stream.latest_result[0] is not None:
                    frame, detections = camera_stream.latest_result[0]
                    camera_stream.latest_result[0] = None

                    new_tracked_faces = {}
                    detected_face_ids = set()

                    for detection in detections:
                        bbox = detection['bbox']
                        name = detection['name']
                        recognized = detection['recognized']

                        matched_face_id = None
                        max_iou = 0
                        
                        for face_id, tracked_face in camera_stream.tracked_faces.items():
                            iou = compute_iou(bbox, tracked_face.bbox)
                            if iou > 0.35 and iou > max_iou:
                                max_iou = iou
                                matched_face_id = face_id

                        if matched_face_id is not None:
                            tracked_face = camera_stream.tracked_faces[matched_face_id]
                            tracked_face.bbox = bbox
                            tracked_face.confidence_count += 1
                            tracked_face.missing_count = 0

                            if tracked_face.confidence_count >= MIN_CONFIDENCE_FRAMES:
                                if tracked_face.recognized != recognized or tracked_face.name != name:
                                    tracked_face.name = name
                                    tracked_face.recognized = recognized
                                    tracked_face.state_duration = 0
                                    tracked_face.current_state_start_time = current_time

                            tracked_face.last_update_time = current_time
                            new_tracked_faces[matched_face_id] = tracked_face
                            detected_face_ids.add(matched_face_id)
                        else:
                            camera_stream.face_id_counter += 1
                            new_face = TrackedFace(camera_stream.face_id_counter, bbox, name, recognized, current_time)
                            new_tracked_faces[camera_stream.face_id_counter] = new_face
                            detected_face_ids.add(camera_stream.face_id_counter)

                    for face_id, tracked_face in camera_stream.tracked_faces.items():
                        if face_id not in detected_face_ids:
                            tracked_face.missing_count += 1
                            tracked_face.confidence_count = max(0, tracked_face.confidence_count - 1)
                            if tracked_face.missing_count < MAX_MISSING_FRAMES:
                                new_tracked_faces[face_id] = tracked_face

                    camera_stream.tracked_faces = new_tracked_faces

                    for tracked_face in camera_stream.tracked_faces.values():
                        if tracked_face.confidence_count >= MIN_CONFIDENCE_FRAMES:
                            if tracked_face.recognized:
                                known_names_set.add(tracked_face.name)
                            else:
                                num_unknown_total += 1

                    if idx == self.current_camera_index:
                        frame_with_detections = draw_detections(frame.copy(), camera_stream.tracked_faces)
                        frame_to_display = frame_with_detections

        if frame_to_display is not None:
            cv2.imshow('Face Detection', frame_to_display)
            cv2.waitKey(1)

        if current_time - self.last_stats_time >= 0.8:
            self.update_stats(num_unknown_total, list(known_names_set))
            
            if self.logger.should_update(current_time, num_unknown_total, known_names_set):
                self.logger.update_log(num_unknown_total, list(known_names_set), current_time)
                
            self.last_stats_time = current_time

        self.root.after(10, self.update_frame)