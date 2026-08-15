import os
import pkgutil
import importlib

def scan_plugins(path, base_import_path):
    
    if not os.path.exists(path): 
        
        return
    
    for _, name, is_pkg in pkgutil.iter_modules([path]):
        if is_pkg: continue 
        full_name = f"{base_import_path}.{name}"
        try:
            importlib.import_module(full_name)
            
        except Exception as e:
            print(f"  [Loader] ❌ Error loading {full_name}: {e}")
            import traceback
            traceback.print_exc()

def setup_env():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    src_dir = os.path.dirname(current_dir)
    
    scan_plugins(os.path.join(src_dir, 'methods', 'watermarks'), 'src.methods.watermarks')
    
    scan_plugins(os.path.join(src_dir, 'methods', 'attacks'), 'src.methods.attacks')
    
    scan_plugins(os.path.join(src_dir, 'methods', 'metrics'), 'src.methods.metrics')
    
    scan_plugins(os.path.join(src_dir, 'data'), 'src.data')
    scan_plugins(os.path.join(src_dir, 'models'), 'src.models')