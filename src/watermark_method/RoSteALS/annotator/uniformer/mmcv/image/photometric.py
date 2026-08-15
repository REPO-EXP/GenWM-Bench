
import cv2
import numpy as np

from ..utils import is_tuple_of
from .colorspace import bgr2gray, gray2bgr

def imnormalize(img, mean, std, to_rgb=True):
    
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)

def imnormalize_(img, mean, std, to_rgb=True):
    
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  
    cv2.subtract(img, mean, img)  
    cv2.multiply(img, stdinv, img)  
    return img

def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  
    cv2.add(img, mean, img)  
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  
    return img

def iminvert(img):
    
    return np.full_like(img, 255) - img

def solarize(img, thr=128):
    
    img = np.where(img < thr, img, 255 - img)
    return img

def posterize(img, bits):
    
    shift = 8 - bits
    img = np.left_shift(np.right_shift(img, shift), shift)
    return img

def adjust_color(img, alpha=1, beta=None, gamma=0):
    
    gray_img = bgr2gray(img)
    gray_img = np.tile(gray_img[..., None], [1, 1, 3])
    if beta is None:
        beta = 1 - alpha
    colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
    if not colored_img.dtype == np.uint8:
        
        colored_img = np.clip(colored_img, 0, 255)
    return colored_img

def imequalize(img):
    
    def _scale_channel(im, c):
        
        im = im[:, :, c]
        
        histo = np.histogram(im, 256, (0, 255))[0]
        
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        if not step:
            lut = np.array(range(256))
        else:
            
            lut = (np.cumsum(histo) + (step // 2)) // step
            
            lut = np.concatenate([[0], lut[:-1]], 0)
            
            lut[lut > 255] = 255
        
        return np.where(np.equal(step, 0), im, lut[im])

    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img.astype(img.dtype)

def adjust_brightness(img, factor=1.):
    
    degenerated = np.zeros_like(img)
    
    brightened_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    brightened_img = np.clip(brightened_img, 0, 255)
    return brightened_img.astype(img.dtype)

def adjust_contrast(img, factor=1.):
    
    gray_img = bgr2gray(img)
    hist = np.histogram(gray_img, 256, (0, 255))[0]
    mean = round(np.sum(gray_img) / np.sum(hist))
    degenerated = (np.ones_like(img[..., 0]) * mean).astype(img.dtype)
    degenerated = gray2bgr(degenerated)
    contrasted_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    contrasted_img = np.clip(contrasted_img, 0, 255)
    return contrasted_img.astype(img.dtype)

def auto_contrast(img, cutoff=0):
    
    def _auto_contrast_channel(im, c, cutoff):
        im = im[:, :, c]
        
        histo = np.histogram(im, 256, (0, 255))[0]
        
        histo_sum = np.cumsum(histo)
        cut_low = histo_sum[-1] * cutoff[0] // 100
        cut_high = histo_sum[-1] - histo_sum[-1] * cutoff[1] // 100
        histo_sum = np.clip(histo_sum, cut_low, cut_high) - cut_low
        histo = np.concatenate([[histo_sum[0]], np.diff(histo_sum)], 0)

        low, high = np.nonzero(histo)[0][0], np.nonzero(histo)[0][-1]
        
        if low >= high:
            return im
        scale = 255.0 / (high - low)
        offset = -low * scale
        lut = np.array(range(256))
        lut = lut * scale + offset
        lut = np.clip(lut, 0, 255)
        return lut[im]

    if isinstance(cutoff, (int, float)):
        cutoff = (cutoff, cutoff)
    else:
        assert isinstance(cutoff, tuple), 'cutoff must be of type int, '            f'float or tuple, but got {type(cutoff)} instead.'
    
    s1 = _auto_contrast_channel(img, 0, cutoff)
    s2 = _auto_contrast_channel(img, 1, cutoff)
    s3 = _auto_contrast_channel(img, 2, cutoff)
    contrasted_img = np.stack([s1, s2, s3], axis=-1)
    return contrasted_img.astype(img.dtype)

def adjust_sharpness(img, factor=1., kernel=None):
    
    if kernel is None:
        
        kernel = np.array([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13
    assert isinstance(kernel, np.ndarray),        f'kernel must be of type np.ndarray, but got {type(kernel)} instead.'
    assert kernel.ndim == 2,        f'kernel must have a dimension of 2, but got {kernel.ndim} instead.'

    degenerated = cv2.filter2D(img, -1, kernel)
    sharpened_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    sharpened_img = np.clip(sharpened_img, 0, 255)
    return sharpened_img.astype(img.dtype)

def adjust_lighting(img, eigval, eigvec, alphastd=0.1, to_rgb=True):
    
    assert isinstance(eigval, np.ndarray) and isinstance(eigvec, np.ndarray),        f'eigval and eigvec should both be of type np.ndarray, got '        f'{type(eigval)} and {type(eigvec)} instead.'

    assert eigval.ndim == 1 and eigvec.ndim == 2
    assert eigvec.shape == (3, eigval.shape[0])
    n_eigval = eigval.shape[0]
    assert isinstance(alphastd, float), 'alphastd should be of type float, '        f'got {type(alphastd)} instead.'

    img = img.copy().astype(np.float32)
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  

    alpha = np.random.normal(0, alphastd, n_eigval)
    alter = eigvec        * np.broadcast_to(alpha.reshape(1, n_eigval), (3, n_eigval))        * np.broadcast_to(eigval.reshape(1, n_eigval), (3, n_eigval))
    alter = np.broadcast_to(alter.sum(axis=1).reshape(1, 1, 3), img.shape)
    img_adjusted = img + alter
    return img_adjusted

def lut_transform(img, lut_table):
    
    assert isinstance(img, np.ndarray)
    assert 0 <= np.min(img) and np.max(img) <= 255
    assert isinstance(lut_table, np.ndarray)
    assert lut_table.shape == (256, )

    return cv2.LUT(np.array(img, dtype=np.uint8), lut_table)

def clahe(img, clip_limit=40.0, tile_grid_size=(8, 8)):
    
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert isinstance(clip_limit, (float, int))
    assert is_tuple_of(tile_grid_size, int)
    assert len(tile_grid_size) == 2

    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    return clahe.apply(np.array(img, dtype=np.uint8))
