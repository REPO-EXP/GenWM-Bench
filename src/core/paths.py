
import os

def get_project_root():
    
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resolve_model_path(path: str) -> str:
    
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(get_project_root(), path))
