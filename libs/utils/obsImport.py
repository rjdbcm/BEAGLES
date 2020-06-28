import cv2
import math
import os
import subprocess
from PyQt5.QtCore import *


class VideoGrid:

    def __init__(self, numVideos, video):
        self.div = math.sqrt(numVideos)
        try:  # make sure the number of camera divisions is always an integer
            assert isinstance(self.div, int)
        except AssertionError:
            self.div = int(math.ceil(self.div))

        self.video = video

        self.numVideos = self.div ** 2

        self.obsFolder = os.path.dirname(video)

        self.w, self.h = self._get_resolution(video)

    @staticmethod
    def _get_resolution(target):
        vid = cv2.VideoCapture(target)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        return width, height

    def crop(self):
        xs = list()
        ys = list()
        h_inc = self.h / self.div
        w_inc = self.w / self.div

        # setup ys
        for i in range(1, self.div + 1):
            ys.append(0)
        for i in range(1, self.div):
            for _ in range(1, self.div + 1):
                ys.append(i * h_inc)
        # setup xs
        for i in range(1, self.div + 1):
            xs.append(0)
            for j in range(1, self.div):
                xs.append(j * w_inc)

        cmd = 'ffmpeg -i "{}" -filter:v "crop={}:{}:{}:{}" -c:a copy "{}"'
        form = '{}_camera_{}{}'
        for i in range(1, self.numVideos + 1):
            name, ext = os.path.splitext(self.video)
            output = form.format(name, i, ext)
            x = xs[i-1]
            y = ys[i-1]
            print('Running: ',
                  cmd.format(self.video, w_inc, h_inc, x, y, output))
            subprocess.Popen(cmd.format(
                self.video, w_inc, h_inc, x, y, output), shell=True)


class OBSConfig:
    '''Methods for writing OBS profiles and scene collections'''
    def __init__(self, numCameras, obsFolder=QDir):
        pass

    def write_profile(self):
        pass

    def write_scene_collection(self):
        pass