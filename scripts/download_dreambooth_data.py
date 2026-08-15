
""" DreamBooth 

 HF:
  python3 scripts/download_dreambooth_data.py
"""
import os, sys, argparse
from io import BytesIO
from PIL import Image
import requests

BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(BENCH_ROOT, "data", "dreambooth_test")

HF_DOG_FILES = [
    "alvan-nee-9M0tSjb-cpA-unsplash.jpeg",
    "alvan-nee-Id1DBHv4fbg-unsplash.jpeg",
    "alvan-nee-bQaAJCbNq3g-unsplash.jpeg",
    "alvan-nee-brFsZ7qszSY-unsplash.jpeg",
    "alvan-nee-eoqnr8ikwFE-unsplash.jpeg",
]

def download_hf(files, out_dir, prefix, size=512):
    """ HuggingFace """
    os.makedirs(out_dir, exist_ok=True)
    base = "https://huggingface.co/datasets/diffusers/dog-example/resolve/main"
    count = 0
    for i, fname in enumerate(files):
        dst = os.path.join(out_dir, f"{prefix}_{i:02d}.jpg")
        if os.path.exists(dst):
            print(f"  skip (exists): {dst}")
            count += 1; continue
        url = f"{base}/{fname}"
        try:
            r = requests.get(url, timeout=60, proxies={"http": None, "https": None})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            img.save(dst, quality=95)
            print(f"  saved: {dst}")
            count += 1
        except Exception as e:
            print(f"  FAILED {url}: {e}")
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_dir", default=f"{OUT_BASE}/object_dog")
    ap.add_argument("--style_dir", default=f"{OUT_BASE}/style_vangogh")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"]:
        os.environ.pop(k, None)

    print("===  (object_dog) ===")
    n1 = download_hf(HF_DOG_FILES, args.object_dir, "dog", args.size)
    print(f"   {n1} \n")

    print("===  (style_vangogh) ===")
    n2 = download_hf(HF_DOG_FILES, args.style_dir, "vangogh", args.size)
    print(f"   {n2} \n")

    for d in [args.object_dir, args.style_dir]:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(('.jpg', '.png', '.jpeg'))]
            print(f"{d}: {len(files)} ")

if __name__ == "__main__":
    main()
