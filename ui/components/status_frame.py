import customtkinter as ctk

class StatusFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.status_label = ctk.CTkLabel(
            self,
            text="System Status: Ready",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=10)
    
    def update_status(self, status):
        self.status_label.configure(text=status)