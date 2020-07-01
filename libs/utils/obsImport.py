import cv2
import math
import os
import subprocess
from .flags import FlagIO


class TiledCaptureArray(FlagIO):
    """Object definition for tiled capture arrays.
    __init__ Args:
            num_divisions: number of tiles to process
            video: path to source video
            unused_cameras: list of camera sources to skip during processing
    Methods:
        crop:
            stream copies processed tiles to labeled files
    """
    def __init__(self, num_divisions: int, video, unused_cameras: list):
        FlagIO.__init__(self, subprogram=True)
        self.div = math.sqrt(num_divisions)
        try:  # make sure the number of camera divisions is always an integer
            assert isinstance(self.div, int)
        except AssertionError:
            self.div = int(math.ceil(self.div))

        self.video = video

        self.num_videos = self.div ** 2

        self.unused_cameras = unused_cameras

        self.folder = os.path.dirname(video)

        self.width, self.height = self._get_resolution(video)

    @staticmethod
    def _get_resolution(target):
        vid = cv2.VideoCapture(target)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        return width, height

    def crop(self):
        xs = list()
        ys = list()
        h_inc = int(self.height / self.div)
        w_inc = int(self.width / self.div)

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

        # open ffmpeg subprocesses
        cmd = 'ffmpeg -y -i "{}" -filter:v "crop={}:{}:{}:{}" -c:a copy "{}"'
        form = '{}_camera_{}{}'
        msg = 'Started ffmpeg PID: {} Output: {}'
        for i in range(1, self.num_videos + 1):
            if i in self.unused_cameras:
                continue
            name, ext = os.path.splitext(self.video)
            output = form.format(name, i, ext)
            x = xs[i-1]
            y = ys[i-1]
            self.logger.debug(cmd.format(self.video, w_inc, h_inc, x, y, output))
            proc = subprocess.Popen(cmd.format(
                self.video, w_inc, h_inc, x, y, output),
                stdout=subprocess.PIPE, shell=True)
            self.logger.info(msg.format(proc.pid, output))
