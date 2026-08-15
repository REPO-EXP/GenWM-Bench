
import numpy as np
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['contour_expand'])

def contour_expand(kernel_mask, internal_kernel_label, min_kernel_area,
                   kernel_num):
    
    assert isinstance(kernel_mask, (torch.Tensor, np.ndarray))
    assert isinstance(internal_kernel_label, (torch.Tensor, np.ndarray))
    assert isinstance(min_kernel_area, int)
    assert isinstance(kernel_num, int)

    if isinstance(kernel_mask, np.ndarray):
        kernel_mask = torch.from_numpy(kernel_mask)
    if isinstance(internal_kernel_label, np.ndarray):
        internal_kernel_label = torch.from_numpy(internal_kernel_label)

    if torch.__version__ == 'parrots':
        if kernel_mask.shape[0] == 0 or internal_kernel_label.shape[0] == 0:
            label = []
        else:
            label = ext_module.contour_expand(
                kernel_mask,
                internal_kernel_label,
                min_kernel_area=min_kernel_area,
                kernel_num=kernel_num)
            label = label.tolist()
    else:
        label = ext_module.contour_expand(kernel_mask, internal_kernel_label,
                                          min_kernel_area, kernel_num)
    return label
