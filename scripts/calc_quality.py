
import os, sys, io, warnings
warnings.filterwarnings('ignore')

_old=sys.stdout; sys.stdout=io.StringIO()
try:
    import lpips, torch
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(script_dir)
    alex_ckpt = os.path.join(proj_root, 'data', 'models', 'Alexnet', 'alexnet-owt-7be5be79.pth')
    if not os.path.exists(alex_ckpt): alex_ckpt = None
    lp_model = lpips.LPIPS(net='alex', pretrained=alex_ckpt).cpu()
except: lp_model=None
sys.stdout=_old

import numpy as np
from PIL import Image
try: from skimage.metrics import structural_similarity as ssim_fn
except: ssim_fn=None
try: from skimage.metrics import peak_signal_noise_ratio as psnr_fn
except: psnr_fn=None

def load_sorted(d):
    if not d or not os.path.isdir(d): return []
    files=sorted(os.listdir(d))
    imgs=[]
    for f in files:
        try: imgs.append(np.array(Image.open(os.path.join(d,f)).convert('RGB').resize((512,512)),dtype=np.float32)/255.)
        except: imgs.append(None)
    return imgs

clean_dir=sys.argv[1] if len(sys.argv)>1 else ''
wm_dir=sys.argv[2] if len(sys.argv)>2 else ''
atk_dir=sys.argv[3] if len(sys.argv)>3 else ''

clean=load_sorted(clean_dir); wm=load_sorted(wm_dir); atk=load_sorted(atk_dir)

sv,pv,lv=[],[],[]
n=min(len(clean), len(wm))
for i in range(n):
    if clean[i] is None or wm[i] is None: continue
    try:
        if ssim_fn: sv.append(ssim_fn(clean[i], wm[i], channel_axis=2, data_range=1.0))
        if psnr_fn: pv.append(psnr_fn(clean[i], wm[i], data_range=1.0))
        if lp_model:
            ct=torch.from_numpy(clean[i]).permute(2,0,1).unsqueeze(0).float()*2-1
            wt=torch.from_numpy(wm[i]).permute(2,0,1).unsqueeze(0).float()*2-1
            lv.append(lp_model(ct,wt).item())
    except: pass

s=f"{sum(sv)/len(sv):.4f}" if sv else "-"
p=f"{sum(pv)/len(pv):.4f}" if pv else "-"
l_=f"{sum(lv)/len(lv):.4f}" if lv else "-"

av=[]
m=min(len(wm), len(atk))
for i in range(m):
    if wm[i] is None or atk[i] is None: continue
    try:
        if psnr_fn: av.append(psnr_fn(wm[i], atk[i], data_range=1.0))
    except: pass

wa=f"{sum(av)/len(av):.4f}" if av else "-"
print(f"{s},{p},{l_},-,{wa}")
