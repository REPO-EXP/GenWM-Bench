
import numpy as np
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['pixel_group'])

def pixel_group(score, mask, embedding, kernel_label, kernel_contour,
                kernel_region_num, distance_threshold):
    
    assert isinstance(score, (torch.Tensor, np.ndarray))
    assert isinstance(mask, (torch.Tensor, np.ndarray))
    assert isinstance(embedding, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_label, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_contour, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_region_num, int)
    assert isinstance(distance_threshold, float)

    if isinstance(score, np.ndarray):
        score = torch.from_numpy(score)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)
    if isinstance(kernel_label, np.ndarray):
        kernel_label = torch.from_numpy(kernel_label)
    if isinstance(kernel_contour, np.ndarray):
        kernel_contour = torch.from_numpy(kernel_contour)

    if torch.__version__ == 'parrots':
        label = ext_module.pixel_group(
            score,
            mask,
            embedding,
            kernel_label,
            kernel_contour,
            kernel_region_num=kernel_region_num,
            distance_threshold=distance_threshold)
        label = label.tolist()
        label = label[0]
        list_index = kernel_region_num
        pixel_assignment = []
        for x in range(kernel_region_num):
            pixel_assignment.append(
                np.array(
                    label[list_index:list_index + int(label[x])],
                    dtype=np.float))
            list_index = list_index + int(label[x])
    else:
        pixel_assignment = ext_module.pixel_group(score, mask, embedding,
                                                  kernel_label, kernel_contour,
                                                  kernel_region_num,
                                                  distance_threshold)
    return pixel_assignment
