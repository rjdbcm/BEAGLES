from unittest import TestCase
from libs.io.flags import FlagIO
from libs.utils.flags import Flags
from libs.net.build import TFNet
from libs.utils.errors import GradientNaN, VariableIsNone
from libs.dark.darknet import Darknet
from libs.net.framework import create_framework
from libs.cythonUtils.nms import iou_c
import tensorflow as tf


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

    def testGradientNan(self):
        with self.assertRaises(GradientNaN):
            raise GradientNaN(self.flags)

    def testFlagIO(self):
        self.io_flags()

    def testCythonExtensions(self):
        iou = iou_c(3, 4, 4, 3, 4, 3, 3, 4)
        self.assertAlmostEqual(0.35211268067359924, iou), "cythonUtils math failure"

    def tearDown(self):
        self.cleanup_ramdisk()
