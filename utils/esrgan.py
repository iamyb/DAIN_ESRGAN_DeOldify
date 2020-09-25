import sys,os
cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(root_path, 'ESRGAN'))

import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from pathlib import Path
import shutil
from .configs import *
from .util import *

def _prepare_images(dain_image_path):
    all_images = list(Path(dain_image_path).rglob('*.png'))
    all_images = sorted(all_images, key=lambda x: int(str(os.path.basename(x)).split('.')[0].split('_')[1]))
    return all_images

def _load_model():
    model_path = './ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    return model

def _esrgan_infer(model, all_images, esrgan_image_path):
    if not os.path.exists(esrgan_image_path):
        os.makedirs(esrgan_image_path)

    device = torch.device('cuda')
    idx = 0
    for path in all_images:
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, path)
        # read images
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('{:s}/{:s}.png'.format(esrgan_image_path, base), output)
    

def esrgan(target, cleanup=True):
    #dain_image_path = os.path.join("./data/frames/dain/", target)
    dain_image_path = get_upstream_path(target, 'esrgan')
    esrgan_image_path = os.path.join("./data/frames/esrgan", target)

    all_images = _prepare_images(dain_image_path)
    model = _load_model()

    _esrgan_infer(model, all_images, esrgan_image_path)
    
    if cleanup:
        shutil.rmtree(dain_image_path)

if __name__ == "__main__":
    esrgan('shanghai1937.mp4')

