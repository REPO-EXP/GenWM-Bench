
import torch
import math
import numpy as np

from .transforms import image_std

def psnr(x, y):
    
    delta = x - y
    delta = 255 * (delta * image_std.view(1, 3, 1, 1).to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  
    peak = 20 * math.log10(255.0)
    noise = torch.mean(delta**2, dim=(1,2,3))  
    psnr = peak - 10*torch.log10(noise)
    return psnr

def iou(preds, targets, threshold=0.0, label=1):
    
    preds = preds > threshold  
    targets = targets > 0.5
    if label == 0:
        preds = ~preds
        targets = ~targets
    intersection = (preds & targets).float().sum((1,2,3))  
    union = (preds | targets).float().sum((1,2,3))  
    
    union[union == 0.0] = intersection[union == 0.0] = 1
    iou = intersection / union
    return iou

def accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.0
) -> torch.Tensor:
    
    preds = preds > threshold  
    targets = targets > 0.5
    correct = (preds == targets).float()  
    accuracy = torch.mean(correct, dim=(1,2,3))  
    return accuracy

def bit_accuracy(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None, threshold: float = 0.0) -> torch.Tensor:
    
    if len(targets.shape) !=3:
        print(f"targets.shape: {targets.shape}")
        targets = targets.unsqueeze(1)
    preds = preds > threshold  
    targets = targets > 0.5  
    correct = (preds.unsqueeze(1) == targets.unsqueeze(-1).unsqueeze(-1)).float()  
    if masks is not None:
        masks = masks.unsqueeze(2)  
        correct = correct * masks  
        bit_acc =  correct.sum() / (masks.sum() * correct.shape[2]) 
    
    return bit_acc

def bit_accuracy_inference(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor,
    method: str = 'hard',
    nb_repetitions: int = 1,
    threshold: float = 0.0
) -> torch.Tensor:
    
    assert preds.shape[1] % nb_repetitions == 0, preds.shape[1] % nb_repetitions
    a = preds.shape[1] // nb_repetitions
    for i in range(nb_repetitions-1):
        preds[:, :a, :, :] += preds[:, (1+i)*a:(i+2)*a, :, :]
    preds = preds[:, :a, :, :]
    targets = targets[:, :a]  

    if method == 'hard':
        
        preds = preds > threshold  
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  
        masks = masks.expand_as(preds).bool()
        
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  
        preds = torch.stack(preds, dim=0)  
    elif method == 'semihard':
        
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  
        masks = masks.expand_as(preds).bool()
        
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  
        preds = torch.stack(preds, dim=0)  
    elif method == 'soft':
        
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  
        preds = torch.sum(preds * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))  
    preds = preds > threshold  
    targets = targets > 0.5  
    correct = (preds == targets).float()  
    bit_acc = torch.mean(correct, dim=(1))  
    return bit_acc

def msg_predict_inference(
    preds: torch.Tensor,
    masks: torch.Tensor,
    method: str = 'semihard',
    threshold: float = 0.0
) -> torch.Tensor:
    
    assert method in ['hard', 'semihard', 'soft'], f"Method {method} not supported"
    if method == 'hard':
        
        preds = preds > threshold  
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  
        masks = masks.expand_as(preds).bool()
        
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  
        preds = torch.stack(preds, dim=0)  
    elif method == 'semihard':
        
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  
        masks = masks.expand_as(preds).bool()
        
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  
        preds = torch.stack(preds, dim=0)  
    elif method == 'soft':
        
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  
        preds = torch.sum(preds * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))  
    preds = preds > 0.5  
    return preds
