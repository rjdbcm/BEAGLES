import cv2
import math
import os
import re
import subprocess
from datetime import datetime
from beagles.io.flags import SharedFlagIO

DATETIME_FORMAT = {
    'underscore': '%Y-%m-%d_%H-%M-%S',
    'space':      '%Y-%m-%d %H-%M-%S'
}
DATETIME_RE = re.compile(r'([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])[_\s][0-1][0-9]-[0-6][0-9]-[0-6][0-9])')


def datetime_from_filename(filename):
    """Extracts a datetime object from OBS Video Filename
       containing %CCYY-%MM-%DD %hh-%mm-%ss or %CCYY-%MM-%DD_%hh-%mm-%ss"""
    return datetime.strptime(DATETIME_RE.search(filename).groups()[0], DATETIME_FORMAT['underscore'])


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
        self.logger = SharedFlagIO().logger
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
                  f'"crop={w_inc}:{h_inc}:{x}:{y}" -c:a copy -map_metadata 0 ' \
                  f'-map_metadata:s:v 0:s:v -map_metadata:s:a 0:s:a "{output}"'
            self.logger.debug(cmd)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            self.logger.info(f'Started ffmpeg PID: {proc.pid} Output: {output}')
