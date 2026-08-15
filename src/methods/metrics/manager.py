import torch
from src.core.registry import METRICS

METRIC_BEHAVIORS = {
    'BasicQualityMetrics': 'fidelity_group', 
    'NIQEMetric':          'no_ref',         
    'PIQEMetric':          'no_ref',
    'CLIPScoreMetric':     'alignment',      
    'DreamSimMetric':      'vs_gt',          
    'FIDMetric':           'distribution',   
}

class MetricManager:
    def __init__(self, config_list, device='cuda'):
        """
        config_list: YAML  metrics :
          [
            {"type": "BasicQualityMetrics"},
            {"type": "FIDMetric", "params": {...}},
            {"type": "NIQEMetric"}
          ]
        """
        self.device = device
        self.active_metrics = {
            'per_image': [],   
            'distribution': [] 
        }
        
        self._load_metrics(config_list)

    def _load_metrics(self, config_list):
        print(f"🔧 ...")
        
        for cfg in config_list:
            
            if isinstance(cfg, str):
                metric_type = cfg
                metric_params = {}
            elif isinstance(cfg, dict):
                metric_type = cfg.get('type')
                metric_params = cfg.copy()
                if 'type' in metric_params: del metric_params['type']
            else:
                continue

            behavior = METRIC_BEHAVIORS.get(metric_type, 'fidelity_group') 

            try:
                
                model = METRICS.build(metric_type, config=metric_params, device=self.device)
                
                print(f"  ✅ Loaded [{metric_type}] -> Mode: {behavior}")
                
                if behavior == 'distribution':
                    self.active_metrics['distribution'].append(model)
                else:
                    
                    self.active_metrics['per_image'].append((model, behavior, metric_type))
                    
            except Exception as e:
                print(f"  ❌ Failed to load {metric_type}: {e}")
                
    def get_per_image_metrics(self):
        return self.active_metrics['per_image']

    def get_distribution_metrics(self):
        return self.active_metrics['distribution']