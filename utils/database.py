import os
import shutil
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_API_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def sync_face_folder():
    local_face_dir = "face"
    
    if os.path.exists(local_face_dir):
        shutil.rmtree(local_face_dir)
    
    os.makedirs(local_face_dir)
    
    try:
        response = supabase.storage.from_('face').list()
        
        for folder in response:
            folder_name = folder['name']
            
            local_subfolder = os.path.join(local_face_dir, folder_name)
            os.makedirs(local_subfolder)
            
            files = supabase.storage.from_('face').list(folder_name)
            
            for file in files:
                if file['name'].lower().endswith('.jpg'):
                    file_path = f"{folder_name}/{file['name']}"
                    
                    data = supabase.storage.from_('face').download(file_path)
                    
                    local_file_path = os.path.join(local_subfolder, file['name'])
                    with open(local_file_path, 'wb') as f:
                        f.write(data)
                    
        print("Đồng bộ folder face thành công!")
        
    except Exception as e:
        print(f"Lỗi khi đồng bộ folder face: {str(e)}")
        if os.path.exists(local_face_dir):
            shutil.rmtree(local_face_dir)
        raise e