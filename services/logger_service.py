from config.settings import *
from supabase import create_client

class FaceDetectionLogger:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.last_stranger_count = 0
        self.last_known_faces = set()
        self.last_update_time = 0
    
    def should_update(self, current_time, stranger_count, known_faces):
        if current_time - self.last_update_time < MIN_LOG_UPDATE_INTERVAL:
            return False
        
        if isinstance(known_faces, list):
            known_faces = set(known_faces)
        
        return (stranger_count != self.last_stranger_count or known_faces != self.last_known_faces)
    
    def update_log(self, stranger_count, known_faces, current_time):
        try:
            log_data = {
                "stranger": stranger_count,
                "face_name": list(known_faces) if isinstance(known_faces, set) else known_faces
            }
            
            self.supabase.table('access_log').insert(log_data).execute()
            
            self.last_stranger_count = stranger_count
            self.last_known_faces = (
                set(known_faces) if isinstance(known_faces, list) 
                else known_faces
            )
            self.last_update_time = current_time
            
            return True
            
        except Exception as e:
            print(f"Error updating access log: {str(e)}")
            return False