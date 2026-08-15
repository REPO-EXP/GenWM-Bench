
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import fftpack
import sys, os
from pathlib import Path
import numpy as np 
import random
import glob
import json
import time
import importlib
import pandas as pd
from tqdm import tqdm

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
cmap = plt.get_cmap("tab10")  
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']  

FONT = '/vol/research/tubui1/_base/utils/FreeSans.ttf'

def make_grid(array_list, gsize=(3,3)):
    
    assert len(gsize)==2 and gsize[0]*gsize[1]==len(array_list)
    h,w,c = array_list[0].shape
    out = np.array(array_list).reshape(gsize[0], gsize[1], h, w, c).transpose(0, 2, 1, 3, 4).reshape(gsize[0]*h, gsize[1]*w, c)
    return out 

def collage(im_list, size=None, pad=0, color=255):
    
    if size is None:
        size=(1, len(im_list))
    assert len(size)==2
    if isinstance(im_list[0], np.ndarray):
        im_list = [Image.fromarray(im) for im in im_list]
    h, w = size
    n = len(im_list)
    canvas = []
    for i in range(h):
        start, end = i*w, min((i+1)*w, n)
        row = combine_horz(im_list[start:end], pad, color)
        canvas.append(row)
    canvas = combine_vert(canvas, pad, color)
    return canvas

def combine_horz(pil_ims, pad=0, c=255):
    
    widths, heights = zip(*(i.size for i in pil_ims))
    total_width = sum(widths) + (len(pil_ims)-1) * pad
    max_height = max(heights)
    color = (c,c,c)
    new_im = Image.new('RGB', (total_width, max_height), color)
    x_offset = 0
    for im in pil_ims:
        new_im.paste(im, (x_offset,0))
        x_offset += (im.size[0] + pad) 
    return new_im

def combine_vert(pil_ims, pad=0, c=255):
    
    widths, heights = zip(*(i.size for i in pil_ims))
    max_width = max(widths)
    total_height = sum(heights) + (len(pil_ims)-1)*pad
    color = (c,c,c)
    new_im = Image.new('RGB', (max_width, total_height), color)
    y_offset = 0
    for im in pil_ims:
        new_im.paste(im, (0,y_offset))
        y_offset += (im.size[1] + pad)
    return new_im 

def make_text_image(img_shape=(100,20), text='hello', font_path=FONT, offset=(0,0), font_size=16):
    
    im = Image.new('RGB', tuple(img_shape), (255,255,255))
    draw = ImageDraw.Draw(im)

    def get_font_size(max_font_size):
        font = ImageFont.truetype(font_path, max_font_size)
        text_size = font.getsize(text)  
        start_w = int((img_shape[0] - text_size[0]) / 2)
        start_h = int((img_shape[1] - text_size[1])/2)
        if start_h <0 or start_w < 0:
            return get_font_size(max_font_size-2)
        else:
            return font, (start_w, start_h)
    font, pos = get_font_size(font_size)
    pos = (pos[0]+offset[0], pos[1]+offset[1])
    draw.text(pos, text, font=font, fill=0)
    return im

def log_scale(array, epsilon=1e-12):
    
    array = np.abs(array)
    array += epsilon  
    array = np.log(array)
    return array

def dct2(array):
    
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array

def idct2(array):
    
    array = fftpack.idct(array, type=2, norm="ortho", axis=0)
    array = fftpack.idct(array, type=2, norm="ortho", axis=1)
    return array

class DCT(object):
    def __init__(self, log=True):
        self.log = log 

    def __call__(self, x):
        x = np.array(x)
        x = dct2(x)
        if self.log:
            x = log_scale(x)
        
        x = np.clip((x - x.min())/(x.max() - x.min()) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(x)

    def __repr__(self):
        s = f'(Discrete Cosine Transform, logarithm={self.log})'
        return self.__class__.__name__ + s