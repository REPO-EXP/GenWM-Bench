
import json
import os
import pandas as pd
from src.core.registry import DATASETS
from src.data.base import BaseDataset

@DATASETS.register("ParquetDataset")
class ParquetDataset(BaseDataset):
    def __init__(self, source, prompt_key="prompt", **kwargs):
        
        self.prompt_key = prompt_key
        super().__init__(source, **kwargs)

    def _load_data(self):
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Parquet file not found: {self.source}")
        
        df = pd.read_parquet(self.source)
        self.data_source = df.to_dict(orient='records')

    def __getitem__(self, idx):
        item = self.data_source[idx]
        prompt = item.get(self.prompt_key, "")
        
        return {
            "prompt": prompt,
            "gt_image": None, 
            "id": idx
        }
