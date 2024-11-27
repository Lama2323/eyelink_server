class FaceDetectionLogger:
    """Class to manage logging of face detection changes to Supabase"""
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.last_stranger_count = 0
        self.last_known_faces = set()
        self.min_update_interval = 3  # Minimum time between updates in seconds
        self.last_update_time = 0
    
    def should_update(self, current_time, stranger_count, known_faces):
        """Check if we should update the log based on changes and time interval"""
        if isinstance(known_faces, list):
            known_faces = set(known_faces)
        
        if current_time - self.last_update_time < self.min_update_interval:
            return False
        
        has_changes = (
            stranger_count != self.last_stranger_count or
            known_faces != self.last_known_faces
        )
        
        return has_changes
    
    def update_log(self, stranger_count, known_faces, current_time):
        """Update access log in Supabase"""
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