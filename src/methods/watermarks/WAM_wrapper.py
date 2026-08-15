
import torch
import numpy as np
import os, sys
from PIL import Image
from typing import Any, Dict, List, Union
from torchvision import transforms

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

def _ensure_wam_imports():
    
    wm_parent = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'watermark_method'))
    if wm_parent not in sys.path:
        sys.path.insert(0, wm_parent)

@WATERMARKS.register("WAM")
class WAMWatermark(BaseWatermark):
    
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)

        self.device = torch.device(global_config.get('device', 'cuda'))
        self.model_path = self.config.get('model_path', 'data/models/WAM/wam.pth')
        self.nbits = self.config.get('nbits', 32)
        self.img_size = self.config.get('img_size', 256)
        self.scaling_w = self.config.get('scaling_w', 1.0)
        self.scaling_i = self.config.get('scaling_i', 1.0)
        self.num_inference_steps = self.config.get('num_inference_steps', 50)
        self.guidance_scale = self.config.get('guidance_scale', 7.5)
        self.model = None
        self._cached_msgs = None

        self.normalize = transforms.Normalize(
            mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD)
        self.unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(self._IMAGENET_MEAN, self._IMAGENET_STD)],
            std=[1.0 / s for s in self._IMAGENET_STD])

    def _build_model(self):
        
        _ensure_wam_imports()

        from WAM.models.wam import Wam
        from WAM.models.embedder import VAEEmbedder
        from WAM.models.extractor import SegmentationExtractor
        from WAM.augmentation.augmenter import Augmenter
        from WAM.modules.msg_processor import MsgProcessor
        from WAM.modules.vae import VAEEncoder, VAEDecoder
        from WAM.modules.vit import ImageEncoderViT
        from WAM.modules.pixel_decoder import PixelDecoder
        from WAM.modules.jnd import JND
        from functools import partial

        CH = 32
        CH_MULT = (1, 1, 1, 2)
        EMBED_DIM = 768
        NUM_HEADS = 12
        WINDOW_SIZE = 8  

        encoder = VAEEncoder(
            ch=CH, out_ch=3, ch_mult=CH_MULT, num_res_blocks=2,
            attn_resolutions=[], dropout=0.0, in_channels=3,
            resolution=256, z_channels=4, double_z=False)
        msg_processor = MsgProcessor(
            nbits=self.nbits, hidden_size=self.nbits * 2,
            msg_processor_type="binary+concat")
        decoder = VAEDecoder(
            ch=CH, out_ch=3, ch_mult=CH_MULT, num_res_blocks=2,
            attn_resolutions=[], dropout=0.0, in_channels=3,
            resolution=256, z_channels=4 + self.nbits * 2,
            tanh_out=True)  
        embedder = VAEEmbedder(encoder, decoder, msg_processor)

        image_encoder = ImageEncoderViT(
            depth=12, embed_dim=EMBED_DIM, img_size=self.img_size,
            mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=NUM_HEADS, patch_size=16, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11], window_size=WINDOW_SIZE, out_chans=EMBED_DIM)
        pixel_decoder = PixelDecoder(embed_dim=EMBED_DIM, nbits=self.nbits)
        detector = SegmentationExtractor(
            image_encoder=image_encoder, pixel_decoder=pixel_decoder)

        augmenter = Augmenter(
            masks={'kind': 'dumb', 'is_training': False},
            augs={}, augs_params={})

        attenuation = JND(in_channels=1, out_channels=3, blue=True,
                          preprocess=self.unnormalize,
                          postprocess=self.normalize)

        model = Wam(
            embedder=embedder, detector=detector, augmenter=augmenter,
            attenuation=attenuation,
            scaling_w=self.scaling_w, scaling_i=self.scaling_i,
            img_size_extractor=self.img_size)

        return model

    def _ensure_model(self):
        if self.model is not None:
            return

        print(f"   [WAM] Loading model (nbits={self.nbits}) ...")
        self.model = self._build_model()

        ckpt_path = resolve_model_path(self.model_path)
        if os.path.exists(ckpt_path):
            print(f"   [WAM] Loading weights from {ckpt_path}")
            sd = torch.load(ckpt_path, map_location='cpu')
            if 'model' in sd:
                sd = sd['model']
            if 'state_dict' in sd:
                sd = sd['state_dict']
            
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                print(f"   [WAM] ⚠️  Missing keys ({len(missing)}): {missing[:3]}...")
            if unexpected:
                print(f"   [WAM] ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        else:
            print(f"   [WAM] ⚠️  Weights not found at {ckpt_path} — model is UNTRAINED")

        self.model.to(self.device).eval()

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        self._ensure_model()

        prompts = prompt if isinstance(prompt, list) else [prompt]
        seeds = kwargs.get('seed', 42)
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(len(prompts))]

        original = kwargs.get('original_image')
        if original is not None:
            clean = [original] if not isinstance(original, list) else list(original)
        else:
            clean = []
            for p, s in zip(prompts, seeds):
                gen = torch.Generator(self.device).manual_seed(s)
                out = pipeline(p, generator=gen,
                               num_inference_steps=self.num_inference_steps,
                               guidance_scale=self.guidance_scale,
                               height=kwargs.get('height', 512),
                               width=kwargs.get('width', 512))
                if hasattr(out, 'images'):
                    clean.append(out.images[0])
                else:
                    clean.append(out[0][0] if isinstance(out, tuple) else out[0])
            else:
                clean.append(out[0][0] if isinstance(out, tuple) else out[0])

        msgs = self._prep_msgs(len(prompts), secret)

        to_tensor = transforms.ToTensor()
        results = []
        for i, img in enumerate(clean):
            if img is None:
                results.append(None)
                continue
            t = to_tensor(img).unsqueeze(0).to(self.device)          
            t_norm = self.normalize(t)                               
            m = msgs[i:i+1] if i < msgs.shape[0] else msgs[:1]
            with torch.no_grad():
                out = self.model.embed(t_norm, m)
            wm_norm = out['imgs_w'][0]                               
            wm = self.unnormalize(wm_norm).clamp(0, 1)
            results.append(transforms.ToPILImage()(wm.cpu()))

        self._cached_msgs = msgs.cpu()
        return results

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_model()
        images = image if isinstance(image, list) else [image]
        to_tensor = transforms.ToTensor()

        if secret is not None:
            gt = self._prep_msgs(1, secret)[0]
        elif self._cached_msgs is not None:
            gt = self._cached_msgs[0]
        else:
            gt = None

        results = []
        for img in images:
            if img is None:
                results.append({'bit_acc': 0.5})
                continue
            if img.mode != 'RGB':
                img = img.convert('RGB')
            t = to_tensor(img).unsqueeze(0).to(self.device)          
            t_norm = self.normalize(t)                               
            with torch.no_grad():
                out = self.model.detect(t_norm)

            preds = out['preds'][0]              
            msg_ch = preds[1:1+self.nbits]       
            scores = msg_ch.sigmoid().mean(dim=(1, 2))  
            bits = (scores > 0.5).float()

            metrics = {'raw_bits': bits.cpu().tolist()}
            if gt is not None:
                n = min(len(bits), len(gt))
                metrics['bit_acc'] = float((bits[:n] == gt[:n].to(bits.device)).float().mean().item())
            else:
                metrics['bit_acc'] = 0.5
            results.append(metrics)

        return results

    def compute_aggregate_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        
        from src.core.interfaces import BaseWatermark
        return BaseWatermark.compute_aggregate_metrics(self, all_results)

    def _prep_msgs(self, bsz: int, secret: Any) -> torch.Tensor:
        
        if secret is None:
            return self.model.get_random_msg(bsz).to(self.device)
        if torch.is_tensor(secret):
            msgs = secret.float().to(self.device)
        elif isinstance(secret, np.ndarray):
            msgs = torch.from_numpy(secret).float().to(self.device)
        elif isinstance(secret, list):
            msgs = torch.tensor(secret, dtype=torch.float32, device=self.device)
        else:
            return self.model.get_random_msg(bsz).to(self.device)
        if msgs.dim() == 1:
            msgs = msgs.unsqueeze(0)
        if msgs.shape[0] < bsz:
            msgs = msgs.repeat((bsz + msgs.shape[0] - 1) // msgs.shape[0], 1)[:bsz]
        return msgs[:, :self.nbits]
