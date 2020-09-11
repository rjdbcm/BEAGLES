import cv2
import math
import os
import subprocess
from libs.io.flags import FlagIO


class TiledCaptureArray:
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
        self.logger = FlagIO(subprogram=True).logger
        # make sure the number of camera divisions is always an integer
        root = math.sqrt(num_divisions)
        self.div = root if isinstance(root, int) else int(math.ceil(root))

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

        for i in range(1, self.num_videos + 1):
            if i in self.unused_cameras:
                continue
            name, ext = os.path.splitext(self.video)
            x = xs[i-1]
            y = ys[i-1]
            output = f'{name}_camera_{i}{ext}'
            cmd = f'ffmpeg -hide_banner -y -i "{self.video}" -filter:v ' \
                  f'"crop={w_inc}:{h_inc}:{x}:{y}" -c:a copy "{output}"'
            self.logger.debug(cmd)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            self.logger.info(f'Started ffmpeg PID: {proc.pid} Output: {output}')
