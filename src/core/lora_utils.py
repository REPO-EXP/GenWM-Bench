""" LoRA merge —  PEFT diffusers API"""
from collections import defaultdict
import torch
from safetensors.torch import load_file as load_safetensors

def _strip_lora_suffix(k: str) -> str:
    """ LoRA .lora.down/up.weight  _lora.down/up.weight"""
    for pat in ('.lora.down.weight', '.lora.up.weight',
                '_lora.down.weight', '_lora.up.weight'):
        if k.endswith(pat):
            return k[:-len(pat)]
    return k

def _group_lora_pairs(lora_dict):
    """ LoRA key  base key  {down, up} """
    pairs = defaultdict(dict)
    for k, v in lora_dict.items():
        if '.lora.down.weight' in k or '_lora.down.weight' in k:
            pairs[_strip_lora_suffix(k)]['down'] = v
        elif '.lora.up.weight' in k or '_lora.up.weight' in k:
            pairs[_strip_lora_suffix(k)]['up'] = v
    return dict(pairs)

def _lora_key_to_unet_key(k: str) -> str:
    """
     LoRA state_dict  key  UNet  key
    :
      unet.down_blocks.0.attentions.0.proj_in.lora.down.weight → down_blocks.0.attentions.0.proj_in.weight
      unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_k_lora.down.weight → down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight
    """
    if k.startswith('unet.'):
        k = k[5:]
    
    k = k.replace('.processor', '')
    return _strip_lora_suffix(k) + '.weight'

def merge_lora(unet, lora, scale=1.0):
    """ LoRA  merge  UNet (W += scale * up @ down)"""
    pairs = _group_lora_pairs(lora)
    sd = unet.state_dict()
    orig = {}
    for base, p in pairs.items():
        target = _lora_key_to_unet_key(base)
        if target not in sd:
            print(f"[lora_utils] SKIP {base} → {target} (not in UNet)")
            continue
        up = p['up'].to(sd[target].device, sd[target].dtype)
        down = p['down'].to(sd[target].device, sd[target].dtype)
        delta = scale * (up @ down)
        param = unet.get_parameter(target).data
        orig[target] = param.detach().clone()
        param.add_(delta)
    return orig

def unmerge_lora(unet, orig):
    """ merge_lora  UNet """
    for k, v in orig.items():
        unet.get_parameter(k).data.copy_(v)

def load_lora_and_merge(unet, src, scale=1.0):
    """/dict  merge  UNet"""
    if isinstance(src, str):
        src = load_safetensors(src)
    return merge_lora(unet, src, scale)
