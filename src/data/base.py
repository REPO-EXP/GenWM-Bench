
import os
import json
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset, ABC):
    def __init__(self, source: str, root_dir: str = None, max_samples: int = None, **kwargs):
        
        self.source = source
        
        self.root_dir = root_dir if root_dir else source
        self.max_samples = max_samples
        self.kwargs = kwargs
        
        self.data_source = []

        self._load_data()

        if self.max_samples is not None:
            self._truncate(self.max_samples)

    def _truncate(self, n):
        
        if hasattr(self.data_source, 'select'):  
            count = min(len(self.data_source), n)
            self.data_source = self.data_source.select(range(count))
        else:  
            self.data_source = self.data_source[:n]
        print(f"    -> [Data] Truncated to first {len(self.data_source)} samples.")

    def _read_image(self, rel_path):
        
        if not rel_path:
            return None
        
        try:
            full_path = os.path.join(self.root_dir, rel_path)
            if os.path.exists(full_path):
                
                return Image.open(full_path).convert('RGB')
            else:
                print(f"    [Warning] Image file not found: {full_path}")
        except Exception as e:
            print(f"    [Warning] Error loading image: {rel_path} ({e})")
        return None

    @abstractmethod
    def _load_data(self):
        
        pass

    @abstractmethod
    def __getitem__(self, idx):
        
        pass

    def __len__(self):
        return len(self.data_source)