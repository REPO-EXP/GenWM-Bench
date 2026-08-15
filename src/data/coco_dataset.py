import os
import json
from .base import BaseDataset
from src.core.registry import DATASETS

@DATASETS.register("COCODataset")
class COCODataset(BaseDataset):
    def __init__(self, source: str, root_dir: str = None, max_samples: int = None, **kwargs):
        super().__init__(source=source, root_dir=root_dir, max_samples=max_samples, **kwargs)

    def _load_data(self):
        """
         COCO  id  filename 
        """
        meta_path = os.path.join(self.source, "meta_data.json")
        self.prompt_key = self.kwargs.get('prompt_key', 'caption')
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"COCO metadata not found at {meta_path}")
            
        with open(meta_path, 'r') as f:
            data = json.load(f)
            
            self.data_source = data.get('annotations', [])
            
            self.image_info_map = {img['id']: img for img in data.get('images', [])}
            
        print(f"    -> [COCO] Loaded {len(self.data_source)} annotations.")
        print(f"    -> [COCO] Loaded {len(self.image_info_map)} image infos.")

    def __getitem__(self, idx):
        
        annotation = self.data_source[idx]
        image_id = annotation['image_id']
        prompt = annotation.get(self.prompt_key, "")
        
        img_info = self.image_info_map.get(image_id)
        if img_info:
            file_name = img_info.get('file_name') 
        else:
            
            file_name = f"{int(image_id):012d}.jpg"

        gt_image = self._read_image(file_name)
        
        return {
            'prompt': prompt,
            'gt_image': gt_image, 
            'id': image_id,
            'filename': file_name 
        }