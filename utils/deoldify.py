# import the necessary packages
import os
import sys
cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(root_path, 'DeOldify'))
#import requests
import ssl

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import convertToJPG

from os import path
import torch

import fastai
from deoldify.visualize import *
from pathlib import Path
import traceback
import shutil
from .util import *
from .configs import *


#torch.backends.cudnn.benchmark=True


#os.environ['CUDA_VISIBLE_DEVICES']='0'

# define a predict function as an endpoint
def process_video():
    video_path = video_colorizer.colorize_from_file_name('shanghai2.mp4' )

def _deoldify_infer(esrgan_image_path, deoldify_image_path):
    video_colorizer = get_video_colorizer()
    video_colorizer.colorize_esrgan_frames(esrgan_image_path, deoldify_image_path)
    

def deoldify(target, cleanup=True):
    #esrgan_image_path = os.path.join("./data/frames/esrgan", target)
    esrgan_image_path = get_upstream_path(target, 'deoldify')
    deoldify_image_path = os.path.join("./data/frames/deoldify", target)

    _deoldify_infer(esrgan_image_path, deoldify_image_path)
    
    if cleanup:
        shutil.rmtree(esrgan_image_path)

if __name__ == '__main__':
    deoldify('shanghai_test.mp4')

