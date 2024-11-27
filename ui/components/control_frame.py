import customtkinter as ctk

class ControlFrame(ctk.CTkFrame):
    def __init__(self, master, callbacks):
        super().__init__(master)
        
        self.refresh_button = ctk.CTkButton(
            self,
            text="Refresh Faces",
            command=callbacks['refresh'],
            font=("Helvetica", 14)
        )
        self.refresh_button.pack(pady=10, padx=20, fill="x")
        
        self.add_camera_button = ctk.CTkButton(
            self,
            text="Add Camera",
            command=callbacks['add_camera'],
            font=("Helvetica", 14)
        )
        self.add_camera_button.pack(pady=10, padx=20, fill="x")
        
        self.remove_camera_button = ctk.CTkButton(
            self,
            text="Remove Camera",
            command=callbacks['remove_camera'],
            font=("Helvetica", 14)
        )
        self.remove_camera_button.pack(pady=10, padx=20, fill="x")
        
        self.nav_frame = self._create_nav_frame(callbacks)
        self.nav_frame.pack(fill="x", pady=10)
    
    def _create_nav_frame(self, callbacks):
        nav_frame = ctk.CTkFrame(self)
        
        prev_button = ctk.CTkButton(
            nav_frame,
            text="Previous",
            command=callbacks['previous'],
            font=("Helvetica", 14)
        )
        prev_button.pack(side="left", padx=5, expand=True)
        
        next_button = ctk.CTkButton(
            nav_frame,
            text="Next",
            command=callbacks['next'],
            font=("Helvetica", 14)
        )
        next_button.pack(side="right", padx=5, expand=True)
        
        return nav_frame