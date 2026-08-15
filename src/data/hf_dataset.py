
from src.data.base import BaseDataset
from src.core.registry import DATASETS

@DATASETS.register("HuggingFaceDataset")
class HuggingFaceDataset(BaseDataset):
    def __init__(self, source, split="train", prompt_key="text", **kwargs):
        """
        Args:
            prompt_key:  ( 'text', 'caption', 'Prompt')
        """
        self.split = split
        self.prompt_key = prompt_key
        
        super().__init__(source=source, **kwargs)

    def _load_data(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(" datasets : pip install datasets")

        print(f"   [HFDataset] Loading {self.source} ({self.split})...")
        try:
            
            self.data_source = load_dataset(self.source, split=self.split)
        except Exception as e:
            print(f"   [Warning] Split '{self.split}' failed, fallback to 'train'. Error: {e}")
            self.data_source = load_dataset(self.source, split="train")

    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        prompt_text = item.get(self.prompt_key, "")
        
        if not prompt_text:
            prompt_text = item.get('prompt', item.get('text', item.get('caption', "")))

        return {
            "prompt": prompt_text,       
            "gt_image": None,            
            "filename": f"hf_{idx}",     
            "id": idx
        }