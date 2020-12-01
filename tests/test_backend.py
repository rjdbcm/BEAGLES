from unittest import TestCase
import os
import sys
import time
from shutil import rmtree
from subprocess import Popen, PIPE
from zipfile import ZipFile
from beagles.io.flags import SharedFlagIO
from beagles.base import BACKEND_ENTRYPOINT, GradientNaN, Flags
from beagles.backend.net import NetBuilder

class TestBackend(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        open('tests/resources/checkpoint', 'w').close()
        with ZipFile('tests/resources/BCCD.v1-resize-416x416.voc.zip', 'r') as f:
            f.extractall('tests/resources/BCCD')
        time.sleep(5)
        cls.maxDiff = 256

    def printer(self, string):
        print(string)
        self.io.log.info(string)

    def setUp(self):
        self.flags = Flags()
        self.io = SharedFlagIO(flags=self.flags, subprogram=False)
        self.printer("Setting up backend unittests...")
        self.io.io_flags()

    def testWrapperYoloV2Train(self):
        self.printer("Testing train and resume from checkpoint...")
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources/ckpt'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.lr = 0.00001
        self.flags.max_lr = 0.0001
        self.flags.step_size_coefficient = 10
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.save = 1
        self.flags.train = True
        self.flags.train = True
        self.io.io_flags()
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)
        self.flags.load = 1
        self.io.io_flags()
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()

    def testWrapperYoloV2Predict(self):
        self.printer("Testing predict method...")
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources/ckpt'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.lr = 0.00001
        self.flags.max_lr = 0.0001
        self.flags.step_size_coefficient = 10
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.save = 1
        self.flags.imgdir = 'tests/resources/BCCD/test'
        self.io.io_flags()
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)

    def testWrapperYoloV2Annotate(self):
        self.printer("Testing annotate method...")
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources/ckpt'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.lr = 0.00001
        self.flags.max_lr = 0.0001
        self.flags.step_size_coefficient = 10
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.save = 1
        self.flags.video = ['tests/resources/test.mp4']
        self.io.io_flags()
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)

    def testRaiseAndSendError(self):
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources/ckpt'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.lr = 100000000.0
        self.flags.max_lr = 100000000.0
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.train = True
        self.flags.started = True
        self.io.io_flags()
        net_builder = NetBuilder(flags=self.flags)
        self.assertRaises(GradientNaN, net_builder.train)
        self.flags = self.io.read_flags()
        self.assertEqual(self.flags.error, GradientNaN().message)

    def tearDown(self) -> None:
        self.io.cleanup_flags()

    @classmethod
    def tearDownClass(cls):
        for f in os.listdir('tests/resources/ckpt'):
            if f.endswith(('.data-00000-of-00001', '.index', '.meta', '.profile')):
                f = os.path.join('tests/resources/ckpt', f)
                os.remove(f)
        try:
            os.remove('tests/resources/ckpt/checkpoint')
        except FileNotFoundError:
            pass
        rmtree('tests/resources/BCCD')
        try:
            rmtree('data/summaries/_test')
        except FileNotFoundError:
            pass
