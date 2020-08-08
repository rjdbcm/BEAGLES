from unittest import TestCase
import os
import sys
from glob import glob
import fnmatch
from shutil import rmtree
from subprocess import Popen, PIPE
from zipfile import ZipFile
from PyQt5.QtCore import QObject
from libs.io.flags import FlagIO
from libs.utils.flags import Flags
from libs.utils.errors import GradientNaN
from libs.cythonUtils.nms import iou_c
from libs.constants import BACKEND_ENTRYPOINT
from libs.widgets.backend import BackendThread


class TestBackend(TestCase, FlagIO):
    @classmethod
    def setUpClass(cls) -> None:
        open('tests/resources/checkpoint', 'w').close()
        with ZipFile('tests/resources/BCCD.v1-resize-416x416.voc.zip', 'r') as f:
            f.extractall('tests/resources/BCCD')

    def setUp(self):
        FlagIO.__init__(self)
        self.flags = Flags()
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.train = True
        self.io_flags()

    def testBackendWrapperTrain(self):
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()

    def testGradientNan(self):
        with self.assertRaises(GradientNaN):
            raise GradientNaN(self.flags)

    def testFlagIO(self):
        self.io_flags()

    def testCythonExtensions(self):
        iou = iou_c(3, 4, 4, 3, 4, 3, 3, 4)
        self.assertAlmostEqual(0.35211268067359924, iou), "cythonUtils math failure"

    def tearDown(self) -> None:
        self.cleanup_ramdisk()

    @classmethod
    def tearDownClass(cls):
        for f in os.listdir('tests/resources'):
            if f.endswith(('.data-00000-of-00001', '.index', '.meta', '.profile')):
                f = os.path.join('tests/resources', f)
                os.remove(f)
        os.remove('tests/resources/checkpoint')
        rmtree('tests/resources/BCCD')
        try:
            rmtree('data/summaries/_test')
        except FileNotFoundError:
            pass
