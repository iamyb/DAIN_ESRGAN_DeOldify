import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(root_path, 'DAIN'))

import time
from torch.autograd import Variable
import torch
import random
import numpy as np
import numpy
import networks
from my_args import args
from scipy.misc import imread, imsave
from AverageMeter import  *
import shutil
from pathlib import Path
from .util import get_upstream_path
from . import configs


def _prepare_images(raw_image_dir):
    all_images = list(Path(raw_image_dir).rglob('*.png'))
    all_images = sorted(all_images, key=lambda x: int(str(os.path.basename(x)).split('.')[0].split('_')[1]))
    all_images_pairs = [(all_images[i], all_images[i+1]) for i in range(len(all_images)-1)]

    return all_images_pairs

def _load_model():
    torch.backends.cudnn.benchmark = True # to speed up the
    model = networks.__dict__[args.netName](channel=args.channels,
                                        filter_size = args.filter_size ,
                                        timestep=args.time_step,
                                        training=False)
    if args.use_cuda:
        model = model.cuda()


    args.SAVED_MODEL = './DAIN/model_weights/best.pth'
    if os.path.exists(args.SAVED_MODEL):
        print("The testing model weight is: " + args.SAVED_MODEL)
        if not args.use_cuda:
            pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(args.SAVED_MODEL)
            # model.load_state_dict(torch.load(args.SAVED_MODEL))

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. release the pretrained dict for saving memory
        pretrained_dict = []
    else:
        print("*****************************************************************")
        print("**** We don't load any trained weights **************************")
        print("*****************************************************************")
        exit(-1)

    model = model.eval() # deploy mode
    return model

def _dain_infer(model, all_images_pairs, dain_image_path):
    use_cuda=args.use_cuda
    save_which=args.save_which
    dtype = args.dtype
    count = 0
    
    if not os.path.exists(dain_image_path):
        os.makedirs(dain_image_path)

    for frame0, frame1 in all_images_pairs: 
        print(frame0, frame1)
        arguments_strFirst =  frame0
        arguments_strSecond = frame1

        X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)
        if not channel == 3:
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
        y_ = y_s[save_which]

        if use_cuda:
            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = y_.data.cpu().numpy()
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            if not isinstance(y_, list):
                y_ = y_.data.numpy()
            else:
                y_ = [item.data.numpy() for item in y_]
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                  intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter]  if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

        timestep = args.time_step
        numFrames = int(1.0 / timestep) - 1
        time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]

        shutil.copy(arguments_strFirst, os.path.join(dain_image_path, "frame_{:0>5d}.png".format(count)))
        count  = count+1
        for item, time_offset in zip(y_, time_offsets):
            arguments_strOut = os.path.join(dain_image_path, "frame_{:0>5d}.png".format(count))
            count = count + 1
            imsave(arguments_strOut, np.round(item).astype(numpy.uint8))

def dain(target, net_name='DAIN_slowmotion', time_step=0.5, cleanup=True):
    args.netName   =  net_name
    args.time_step =  time_step
    configs.FPS_MUL= 1/time_step
    #raw_image_path = os.path.join("./data/frames/raw/", target)
    raw_image_path  = get_upstream_path(target, 'dain')
    dain_image_path = os.path.join("./data/frames/dain/", target)

    all_images_pairs = _prepare_images(raw_image_path)
    model = _load_model()

    _dain_infer(model, all_images_pairs, dain_image_path)
    if cleanup:
        shutil.rmtree(raw_image_path)

if __name__ == "__main__":
    dain('shanghai_test.mp4')


         
