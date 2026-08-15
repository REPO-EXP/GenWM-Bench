import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    decoder = torch.load(args.model)
    decoder.eval()
    if args.cuda:
        decoder = decoder.cuda()

    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = ImageOps.fit(image, size)
            image = to_tensor(image).unsqueeze(0)
            if args.cuda:
                image = image.cuda()

            secret = decoder(image)
            if args.cuda:
                secret = secret.cpu()
            secret = np.array(secret[0])
            secret = np.round(secret)
            
            sec=np.array(np.load('secret.npy'))
            print(secret[0:20])
            
            wm = np.array([1 if x else 0 for x in secret])
            Sec= np.array([1 if x else 0 for x in sec])
            error_bits = np.sum(Sec != wm)
            print(1-error_bits/len(wm))
            
if __name__ == "__main__":
    main()
