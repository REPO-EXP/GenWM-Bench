
import numpy as np
import torch
from scipy import linalg
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape,        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape,        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

class SIFID(object):
    def __init__(self, dims=64,device=torch.device("cuda")) -> None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).cuda(device)
        self.model.eval()
        self.dims = dims
        self.device = device
    
    def calculate_activation_statistics(self, x):
        act = self.get_activations(x)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    def get_activations(self, x):
        
        batch_size = x.shape[0]
        with torch.no_grad():
            pred = self.model(x)[0]
            pred = pred.cpu().numpy()
            pred = pred.transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)
        return pred

    def __call__(self, x1, x2):
        
        x1, x2 = (x1 + 1.)/2, (x2 + 1.)/2  
        m1, s1 = self.calculate_activation_statistics(x1.unsqueeze(0).cuda(self.device))
        m2, s2 = self.calculate_activation_statistics(x2.unsqueeze(0).cuda(self.device))
        return calculate_frechet_distance(m1, s1, m2, s2)

class InceptionV3(nn.Module):
    
    DEFAULT_BLOCK_INDEX = 3

    BLOCK_INDEX_BY_DIM = {
        64: 0,   
        192: 1,  
        768: 2,  
        2048: 3  
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=False,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3,            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = torchvision.models.inception_v3(pretrained=True)

        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            ]

        self.blocks.append(nn.Sequential(*block0))

        if self.last_needed_block >= 1:
            block1 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
            ]
            self.blocks.append(nn.Sequential(*block1))

        if self.last_needed_block >= 2:
            block2 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ]
            self.blocks.append(nn.Sequential(*block3))

        if self.last_needed_block >= 4:
            block4 = [
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block4))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
