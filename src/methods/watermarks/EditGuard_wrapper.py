"""
EditGuard wrapper — follows WAM/WTMV2 post-hoc pattern.
Uses EditGuard's native option.parse + create_model + load_test.
"""
import torch
import numpy as np
import os, sys
from PIL import Image
from typing import Any, Dict, List, Union

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

def _ensure_editguard_imports():
    ed_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'watermark_method', 'EditGuard', 'code'))
    if ed_root not in sys.path:
        sys.path.insert(0, ed_root)

@WATERMARKS.register("EditGuard")
class EditGuardWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        self.device = torch.device(global_config.get('device', 'cuda'))
        self.nbits = self.config.get('nbits', 64)
        self.img_size = self.config.get('img_size', 512)
        self.model = None
        self.dwt_fn = None
        self.iwt_fn = None
        self.quant_fn = None
        self._cached_msgs = None

    def _load_model(self):
        if self.model is not None:
            return

        _ensure_editguard_imports()
        from options import options as eg_opt
        from models import create_model
        from models.networks import DWT as _DWT, IWT as _IWT
        from models.modules.Quantization import Quantization as _Quant

        yml = os.path.join(os.path.dirname(__file__), '..', '..',
                           'watermark_method', 'EditGuard', 'code',
                           'options', 'test_editguard.yml')
        ckpt_path = resolve_model_path(
            self.config.get('ckpt_path', 'data/models/EditGuard/clean.pth'))

        opt = eg_opt.parse(yml, is_train=False)
        opt['dist'] = False
        opt['hide'] = True
        opt['message_length'] = self.nbits
        opt['mode'] = 'image'
        opt['gpu_ids'] = [0]
        opt['test'] = {}

        self.model = create_model(opt)
        if isinstance(self.model.netG, torch.nn.DataParallel):
            self.model.netG = self.model.netG.module
        self.model.load_test(ckpt_path)
        self.model.netG.to(self.device).eval()
        self.model.mode = 'image'
        self.model.gop = 1
        self.model.device = self.device

        self.dwt_fn = _DWT()
        self.iwt_fn = _IWT()
        self.quant_fn = _Quant()
        print("   [EditGuard] Ready.")
        torch.cuda.empty_cache()

    def embed(self, pipeline, prompt, secret, **kwargs):
        self._load_model()
        original = kwargs.get('original_image')
        if original is not None:
            clean = [original] if not isinstance(original, list) else list(original)
        else:
            raise ValueError("EditGuard requires 'original_image'")

        msg = self._prep_msg(secret).to(self.device)
        results = []

        blue = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 255))

        for img in clean:
            
            h_np = np.array(img.convert('RGB').resize((self.img_size, self.img_size))).astype(np.float32) / 255.0
            h_t = torch.from_numpy(h_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)

            s_np = np.array(blue).astype(np.float32) / 255.0
            s_t = torch.from_numpy(s_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)

            B, T, C, H, W = h_t.shape
            host_c = h_t[:, T//2:T//2+1]                            
            secret_c = s_t[:, :, T//2:T//2+1]                       

            x_fwd = self.dwt_fn(host_c.reshape(B, -1, H, W))        
            x_h_fwd = [self.dwt_fn(secret_c[:, i].reshape(B, -1, H, W)) for i in range(s_t.shape[1])]

            m = (msg.unsqueeze(0) * 2 - 1) * 0.5                    

            with torch.no_grad():
                _, container = self.model.netG(x=x_fwd, x_h=x_h_fwd, message=m)

            cnt_np = container[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            results.append(Image.fromarray((cnt_np * 255).astype(np.uint8)))

        self._cached_msgs = msg.cpu()
        torch.cuda.empty_cache()
        return results

    def extract(self, image, secret=None, **kwargs):
        self._load_model()
        images = image if isinstance(image, list) else [image]

        if secret is not None:
            gt = self._prep_msg(secret).to(self.device)
        elif self._cached_msgs is not None:
            gt = self._cached_msgs.to(self.device)
        else:
            gt = None

        results = []
        for img in images:
            if img is None:
                results.append({'bit_acc': 0.5})
                continue
            if img.mode != 'RGB':
                img = img.convert('RGB')

            a = np.array(img.resize((self.img_size, self.img_size))).astype(np.float32) / 255.0
            t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)
            B, T, C, H, W = t.shape
            container_c = t[:, T//2:T//2+1]

            y = self.quant_fn(container_c.squeeze(1))

            with torch.no_grad():
                _, _, _, recmessage = self.model.netG(x=y, rev=True)

            rec_bits = (recmessage > 0).float()                     
            gt_bits = ((gt.unsqueeze(0) * 2 - 1) > 0).float()
            bit_acc = float((rec_bits == gt_bits).float().mean().item())

            results.append({'bit_acc': bit_acc})

        return results

    def compute_aggregate_metrics(self, all_results):
        return BaseWatermark.compute_aggregate_metrics(self, all_results)

    def _prep_msg(self, secret):
        if torch.is_tensor(secret):
            m = secret.float().to(self.device)
        elif isinstance(secret, np.ndarray):
            m = torch.from_numpy(secret).float().to(self.device)
        elif isinstance(secret, list):
            m = torch.tensor(secret, dtype=torch.float32, device=self.device)
        else:
            return torch.randint(0, 2, (self.nbits,)).float().to(self.device)
        if m.dim() == 2:
            m = m[0]
        if len(m) < self.nbits:
            m = torch.cat([m, torch.zeros(self.nbits - len(m)).to(self.device)])
        return m[:self.nbits]
