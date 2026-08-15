import torch
import os
from PIL import Image
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

try:
    
    import src.eval_metrics.open_clip as open_clip
except ImportError:
    try:
        
        import open_clip
    except ImportError:
        open_clip = None
        print("[Warning] open_clip not found in 'src.eval_metrics.open_clip' or global packages.")

@METRICS.register("CLIPScoreMetric")
class CLIPScoreMetric(BaseMetric):
    def __init__(self, config=None, **kwargs):
        if config is None: config = {}
        config.update(kwargs)
        super().__init__(config, **kwargs)
        
        if open_clip is None:
            self.model = None
            print("[Error] CLIPScoreMetric cannot run because open_clip is not imported.")
            return

        self.model_name = self.config.get('model_name', 'ViT-B-32')
        
        self.pretrained = self.config.get('pretrained')
        
        self.precision = self.config.get('precision', 'fp32')
        
        print(f"[Metrics] Loading OpenCLIP model: {self.model_name}")
        if self.pretrained:
            print(f"          Pretrained path: {self.pretrained}")

        try:
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained, 
                device=self.device,
                precision=self.precision
            )
            
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            self.model.eval()
            
        except Exception as e:
            print(f"[Error] Failed to load OpenCLIP model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def calculate(self, **kwargs) -> dict:
        if self.model is None: return {}
        
        img = kwargs.get('img_gen_wm')
        prompt = kwargs.get('prompt')
        
        if img is None or not prompt:
            return {}
        
        try:
            
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            
            text_input = self.tokenizer([prompt]).to(self.device)
            
            with torch.no_grad():
                
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                score = (image_features @ text_features.T).item()
                
            return {"clip_score": score}
            
        except Exception as e:
            print(f"[Metrics] CLIP calc error: {e}")
            return {}