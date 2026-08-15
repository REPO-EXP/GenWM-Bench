import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch
from scipy.stats import binom

class BaseWatermark(ABC):
    def __init__(self, config: Dict, global_config: Dict = None):
        self.config = config
        self.global_config = global_config or {}

    @abstractmethod
    def embed(self, pipeline, prompt: str, secret: Any, **kwargs) -> Image.Image:
        pass

    @abstractmethod
    def extract(self, image: Image.Image, secret: Any = None, **kwargs) -> Dict[str, Any]:
        pass

    def _calc_bit_based_tpr(self, bit_accs: List[float], secret_len: int, fpr_target: float) -> float:
        
        threshold_count = binom.isf(fpr_target, secret_len, 0.5)
        print(f"Calculated threshold for TPR@{fpr_target}: {threshold_count:.2f} bits out of {secret_len}")
        print(f"Equivalent Bit Accuracy Threshold: {threshold_count / secret_len:.4f}")
        
        match_counts = np.array(bit_accs) * secret_len
        
        passed = np.sum(match_counts >= threshold_count)
        tpr = passed / len(bit_accs)
        return tpr

    def compute_aggregate_metrics(self, all_sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        
        if not all_sample_results: return {}
        
        metrics = {}
        
        keys = all_sample_results[0].keys()
        for key in keys:
            values = [res[key] for res in all_sample_results if isinstance(res.get(key), (int, float))]
            if values:
                metrics[f"avg_{key}"] = float(np.mean(values))
        
        if 'bit_acc' in keys:
            bit_accs = [res['bit_acc'] for res in all_sample_results]
            
            secret_len = self.config.get('secret_length', 32)
            if 'raw_bits' in all_sample_results[0]:
                secret_len = len(all_sample_results[0]['raw_bits'])
            
            metrics['TPR@1e-2'] = self._calc_bit_based_tpr(bit_accs, secret_len, 1e-2)
            metrics['TPR@1e-6'] = self._calc_bit_based_tpr(bit_accs, secret_len, 1e-6)
            
        return metrics

class BaseAttack(ABC):
    def __init__(self, **kwargs):
        
        self.config = kwargs

    @abstractmethod
    def apply(self, image: Image.Image) -> Image.Image:
        
        pass
    
    def get_param_str(self) -> str:
        
        return self.__class__.__name__

class BaseDataset(ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        
        pass

class BaseModel(ABC):
    def __init__(self, config: Dict, global_config: Dict = None):
        
        self.config = config
        self.global_config = global_config or {}
        self.device = self.global_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.is_loaded = False

    @abstractmethod
    def load(self):
        
        pass

    def get_pipeline(self):
        
        if not self.is_loaded:
            self.load()
        return self.pipeline

    def get_component(self, name: str):
        
        if not self.is_loaded:
            self.load()
        
        if hasattr(self.pipeline, name):
            return getattr(self.pipeline, name)
        else:
            raise ValueError(f"Component {name} not found in pipeline.")
        
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import torch
import numpy as np
from PIL import Image

class BaseMetric(ABC):
    
    def __init__(self, config: Dict, **kwargs):
        self.config = config
        
        self.device = kwargs.get('device', config.get('device', 'cuda'))

    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, float]:
        
        pass

    def compute_aggregate(self, context: Dict) -> Dict[str, float]:
        
        return {}
