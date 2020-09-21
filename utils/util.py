import os
import cv2
from .configs import *

def _extract_raw_frames(videoName):
    sourcePath = os.path.join(INPUT_VIDEO_ROOT, videoName)
    targetPath = OUTPUT_FRAMES_RAW_ROOT
    
    targetPath = os.path.join(targetPath, videoName)
    if os.path.exists(targetPath):
        #shutil.rmtree(targetPath)
        return targetPath

    os.makedirs(targetPath)
    print(sourcePath)

    vidcap = cv2.VideoCapture(sourcePath)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(os.path.join(targetPath, "frame_%.5d.png") % count, image)          
      success,image = vidcap.read()
      print('Read a new frame: %d' % count)
      count += 1

    return targetPath

def get_upstream_path(target, proc):
    chains  = {'raw':'raw', 'dain':'raw', 'esrgan':'dain', 
              'deoldify':'esrgan', 'build': 'deoldify'}
    streams = {'raw'     : OUTPUT_FRAMES_RAW_ROOT,
               'dain'    : OUTPUT_FRAMES_DAIN_ROOT,
               'esrgan'  : OUTPUT_FRAMES_ESRGAN_ROOT,
               'deoldify': OUTPUT_FRAMES_DEOLDIFY_ROOT}

    upstream_path = os.path.join(streams[chains[proc]], target)
    while not os.path.exists(upstream_path) and proc != 'raw':
        proc = chains[proc]
        upstream_path = os.path.join(streams[chains[proc]], target)
        
    if proc == 'raw':
        upstream_path = _extract_raw_frames(target)

    return upstream_path

