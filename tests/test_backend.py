from unittest import TestCase
from libs.utils.flags import FlagIO, Flags
from libs.net.build import TFNet
from libs.dark.darknet import Darknet
from libs.net.framework import create_framework
from libs.cython_utils.nms import box_iou


class TestBackend(TestCase, FlagIO):
    def setUp(self):
        FlagIO.__init__(self)
        self.flags = Flags()
        self.send_flags()
        self.flags.model = 'data/cfg/tiny-yolov2.cfg'
        self.flags.labels = 'tests/test_classes.txt'

    def testDarknet(self):
        self.darknet = Darknet(self.flags)
        create_framework(self.darknet.meta, self.flags)

    def testFlagIO(self):
        self.io_flags()

    def testCythonExtensions(self):
        iou = box_iou(3, 4, 4, 3, 4, 3, 3, 4)
        self.assertAlmostEqual(0.35211268067359924, iou), "cython_utils math failure"

    def tearDown(self):
        self.cleanup_ramdisk()
