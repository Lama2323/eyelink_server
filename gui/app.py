import cv2
import threading
import time
import customtkinter as ctk
from utils.detection import load_face_recognition, detect_faces, draw_detections, face_recognition_data
from utils.database import sync_face_folder
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
        self.fps = 0.0
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
        self.root.geometry("400x700")

        self._camera_sources = []
        self.logger = FaceDetectionLogger()
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Face Detection Control Panel",
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

        self.fps_label = ctk.CTkLabel(
            self.stats_frame,
            text="FPS: ...",
            font=("Helvetica", 16)
        )
        self.fps_label.pack(pady=10)

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
            text="System Status: Ready",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.camera_streams = []
        self.last_stats_time = time.time()
        self.current_camera_index = 0

        self.init_face_recognition()

    def init_face_recognition(self):
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

    def refresh_faces(self):
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
            print("No camera source provided. Exiting.")
            exit()

        self._camera_sources.append(camera_source)
        camera_id = len(self.camera_streams) + 1
        first_camera = CameraStream(camera_source, camera_id)
        if first_camera.start():
            self.camera_streams.append(first_camera)
        else:
            print(f"Failed to start stream {camera_source}")
            exit()

    def add_camera(self, preset_source=None):
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

    def update_stats(self, num_strangers, known_names, fps):
        self.stranger_label.configure(text=f"Stranger: {num_strangers}")
        self.familiar_label.configure(text=f"Familiar face: {', '.join(known_names)}")
        if len(self.camera_streams) > 0:
            average_fps = sum(stream.fps for stream in self.camera_streams) / len(self.camera_streams)
            self.fps_label.configure(text=f"FPS: {average_fps:.2f}")

    def on_closing(self):
        for camera_stream in self.camera_streams:
            camera_stream.stop()
        cv2.destroyAllWindows()
        self.root.destroy()

    def update_frame(self):
        current_time = time.time()
        total_tracked_faces = {}
        num_unknown_total = 0
        known_names_set = set()
        frame_to_display = None

        for idx, camera_stream in enumerate(self.camera_streams):
            with camera_stream.result_lock:
                if camera_stream.latest_result[0] is not None:
                    frame, detections = camera_stream.latest_result[0]
                    camera_stream.latest_result[0] = None

                    new_tracked_faces = {}
                    for detection in detections:
                        bbox = detection['bbox']
                        name = detection['name']
                        recognized = detection['recognized']

                        matched_face_id = None
                        max_iou = 0
                        for face_id, tracked_face in camera_stream.tracked_faces.items():
                            iou = compute_iou(bbox, tracked_face.bbox)
                            if iou > 0.5 and iou > max_iou:
                                max_iou = iou
                                matched_face_id = face_id

                        if matched_face_id is not None:
                            tracked_face = camera_stream.tracked_faces[matched_face_id]
                            tracked_face.bbox = bbox
                            time_since_last_update = current_time - tracked_face.last_update_time

                            if tracked_face.recognized == recognized and tracked_face.name == name:
                                tracked_face.state_duration += time_since_last_update
                            else:
                                if tracked_face.recognized and not recognized:
                                    tracked_face.unknown_duration += time_since_last_update
                                    if tracked_face.unknown_duration >= 3:
                                        tracked_face.name = name
                                        tracked_face.recognized = False
                                        tracked_face.state_duration = 0
                                        tracked_face.current_state_start_time = current_time
                                        tracked_face.unknown_duration = 0
                                else:
                                    tracked_face.name = name
                                    tracked_face.recognized = recognized
                                    tracked_face.state_duration = 0
                                    tracked_face.current_state_start_time = current_time
                                    tracked_face.unknown_duration = 0

                            tracked_face.last_update_time = current_time
                            new_tracked_faces[matched_face_id] = tracked_face
                        else:
                            camera_stream.face_id_counter += 1
                            new_tracked_faces[camera_stream.face_id_counter] = TrackedFace(
                                camera_stream.face_id_counter, bbox, name, recognized, current_time
                            )

                    camera_stream.tracked_faces = {
                        face_id: face for face_id, face in new_tracked_faces.items()
                        if current_time - face.last_update_time <= 5
                    }

                    for tracked_face in camera_stream.tracked_faces.values():
                        if tracked_face.recognized:
                            known_names_set.add(tracked_face.name)
                        else:
                            num_unknown_total += 1

                    camera_stream.frame_count += 1
                    elapsed_time = current_time - camera_stream.start_time
                    if elapsed_time > 1.0:
                        camera_stream.fps = camera_stream.frame_count / elapsed_time
                        camera_stream.frame_count = 0
                        camera_stream.start_time = current_time

                    if idx == self.current_camera_index:
                        frame_with_detections = draw_detections(frame.copy(), camera_stream.tracked_faces)
                        frame_to_display = frame_with_detections

        if frame_to_display is not None:
            cv2.imshow('Face Detection', frame_to_display)
            cv2.waitKey(1)

        if current_time - self.last_stats_time >= 1:
            self.update_stats(num_unknown_total, list(known_names_set), None)

            if self.logger.should_update(current_time, num_unknown_total, known_names_set):
                self.logger.update_log(num_unknown_total, list(known_names_set), current_time)

            self.last_stats_time = current_time

        self.root.after(10, self.update_frame)