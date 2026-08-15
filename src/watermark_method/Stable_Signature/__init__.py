from .utils_model import load_model_from_config
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)