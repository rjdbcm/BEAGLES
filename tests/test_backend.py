from unittest import TestCase
from libs.utils.flags import FlagIO, Flags
from libs.net.build import TFNet
from libs.cython_utils.nms import box_iou


class TestBackend(TestCase, FlagIO):
    def setUp(self):
        FlagIO.__init__(self)
        self.flags = Flags()
        self.send_flags()

    def testFlagIO(self):
        self.io_flags()

    def testCythonExtensions(self):
        iou = box_iou(3, 4, 4, 3, 4, 3, 3, 4)
        self.assertEqual(0.35211268067359924, iou), "cython_utils math failure"

    # def testTFNet(self):
    #     self.tfnet = TFNet(self.flags)

    def tearDown(self):
        self.cleanup_ramdisk()
