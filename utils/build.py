import cv2
import os, sys
import shutil
import ffmpeg
from .util import *
#from .configs import *
from . import configs

def _get_fps(source_path) -> str:
    probe = ffmpeg.probe(str(os.path.join(configs.INPUT_VIDEO_ROOT, source_path)))
    stream_data = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None,
    )
    return (eval(stream_data['avg_frame_rate']))
    #return ((stream_data['avg_frame_rate']).split('/')[0])

def build(videoName, cleanup=True):
    videoPath = os.path.join(configs.OUTPUT_VIDEO_ROOT, videoName)
    if os.path.exists(videoPath):
        os.remove(videoPath)

    #fps = int(_get_fps(videoName)) * mul
    mul = configs.FPS_MUL
    fps = (_get_fps(videoName)) * mul
    print(fps)
    upstream_path = get_upstream_path(videoName, 'build')
    template = os.path.join(upstream_path, 'frame_%5d.png')
    ffmpeg.input(str(template), format='image2', framerate=fps).output(videoPath,
            crf = 17, vcodec = 'libx264', pix_fmt = 'yuv420p').run(capture_stdout=True)

    if cleanup:
        shutil.rmtree(upstream_path)
