import customtkinter as ctk

class StatsFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.stranger_label = ctk.CTkLabel(
            self,
            text="Stranger: 0",
            font=("Helvetica", 16)
        )
        self.stranger_label.pack(pady=10)
        
        self.familiar_label = ctk.CTkLabel(
            self,
            text="Familiar face: ",
            font=("Helvetica", 16)
        )
        self.familiar_label.pack(pady=10)
        
        self.fps_label = ctk.CTkLabel(
            self,
            text="FPS: ...",
            font=("Helvetica", 16)
        )
        self.fps_label.pack(pady=10)
    
    def update_stats(self, num_strangers, known_names, fps):
        self.stranger_label.configure(text=f"Stranger: {num_strangers}")
        self.familiar_label.configure(text=f"Familiar face: {', '.join(known_names)}")
        if fps is not None:
            self.fps_label.configure(text=f"FPS: {fps:.2f}")