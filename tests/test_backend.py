from unittest import TestCase
import os
import sys
from shutil import rmtree
from subprocess import Popen, PIPE
from zipfile import ZipFile
from libs.io.flags import FlagIO
from libs.utils.flags import Flags
from libs.backend.net.trainer import Trainer
from libs.utils.errors import GradientNaN
from libs.constants import BACKEND_ENTRYPOINT


class TestBackend(TestCase, FlagIO):
    @classmethod
    def setUpClass(cls) -> None:
        open('tests/resources/checkpoint', 'w').close()
        with ZipFile('tests/resources/BCCD.v1-resize-416x416.voc.zip', 'r') as f:
            f.extractall('tests/resources/BCCD')

    def setUp(self):
        FlagIO.__init__(self)
        self.flags = Flags()

    def testBackendWrapperTrainYoloV2(self):
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.lr = .001
        self.flags.max_lr = .001
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.train = True
        self.io_flags()
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)
        self.flags.load = 63
        self.io_flags()
        proc = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)

    def testBackendGradientExplosion(self):
        self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
        self.flags.dataset = 'tests/resources/BCCD/train'
        self.flags.labels = 'tests/resources/BCCD.classes'
        self.flags.annotation = 'tests/resources/BCCD/train'
        self.flags.backup = 'tests/resources'
        self.flags.project_name = '_test'
        self.flags.trainer = 'adam'
        self.flags.lr = 1000.0
        self.flags.max_lr = 10000.0
        self.flags.load = 0
        self.flags.batch = 4
        self.flags.epoch = 1
        self.flags.train = True
        t = Trainer(self.flags)
        self.assertRaises(GradientNaN, t.train)

    # def testBackendWrapperTrainYoloV1(self):
    #     self.flags.model = 'tests/resources/yolov2-lite-3c.cfg'
    #     self.flags.dataset = 'tests/resources/BCCD/train'
    #     self.flags.labels = 'tests/resources/BCCD.classes'
    #     self.flags.annotation = 'tests/resources/BCCD/train'
    #     self.flags.backup = 'tests/resources'
    #     self.flags.project_name = '_test'
    #     self.flags.trainer = 'adam'
    #     self.flags.load = 0
    #     self.flags.batch = 4
    #     self.flags.epoch = 1
    #     self.flags.train = True
    #     self.io_flags()

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
