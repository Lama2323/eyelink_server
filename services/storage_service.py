import os
import shutil
from config.settings import *
from supabase import create_client

class StorageService:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.face_folder = 'face'
    
    def sync_faces(self):
        if os.path.exists(self.face_folder):
            shutil.rmtree(self.face_folder)
        os.makedirs(self.face_folder)
        
        try:
            response = self.supabase.storage.from_('face').list()
            
            for folder in response:
                folder_name = folder['name']
                local_subfolder = os.path.join(self.face_folder, folder_name)
                os.makedirs(local_subfolder)
                
                files = self.supabase.storage.from_('face').list(folder_name)
                
                for file in files:
                    if file['name'].lower().endswith('.jpg'):
                        file_path = f"{folder_name}/{file['name']}"
                        data = self.supabase.storage.from_('face').download(file_path)
                        
                        local_file_path = os.path.join(local_subfolder, file['name'])
                        with open(local_file_path, 'wb') as f:
                            f.write(data)
            
            return True
            
        except Exception as e:
            if os.path.exists(self.face_folder):
                shutil.rmtree(self.face_folder)
            raise e