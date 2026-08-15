
import numpy as np

import annotator.uniformer.mmcv as mmcv

try:
    import torch
except ImportError:
    torch = None

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    
    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs
