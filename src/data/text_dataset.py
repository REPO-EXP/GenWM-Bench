import os
from src.data.base import BaseDataset
from src.core.registry import DATASETS

@DATASETS.register("TxtFileDataset")
class TxtFileDataset(BaseDataset):
    def __init__(self, source: str, **kwargs):
        """
         txt  prompt
        
        Args:
            source: txt  (e.g., 'data/prompts.txt')
            **kwargs:  BaseDataset  ( max_samples)
        """
        
        super().__init__(source=source, **kwargs)

    def _load_data(self):
        """
         txt 
        """
        print(f"   [TxtDataset] Loading prompts from: {self.source}")
        
        if not os.path.exists(self.source):
            raise FileNotFoundError(f": {self.source}")

        with open(self.source, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.data_source = [line.strip() for line in lines if line.strip()]
        
        print(f"   [TxtDataset] Loaded {len(self.data_source)} prompts.")

    def __getitem__(self, idx):
        """
        
        """
        prompt_text = self.data_source[idx]

        return {
            "prompt": prompt_text,       
            "gt_image": None,            
            "filename": f"line_{idx}",   
            "id": idx                    
        }