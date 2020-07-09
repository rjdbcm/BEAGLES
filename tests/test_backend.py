from unittest import TestCase
from libs.utils.flags import FlagIO, Flags
from libs.cython_utils.nms import box_iou


class TestBackend(TestCase):
    def testRAMDiskAndFlagIO(self):
        self.io = FlagIO()
        self.io.flags = Flags()
        self.io.init_ramdisk()
        self.io.io_flags()
        self.io.cleanup_ramdisk()

    def testCythonExtensions(self):
        iou = box_iou(3,4,4,3,4,3,3,4)
        self.assertEqual(0.35211268067359924, iou), "cython_utils math failure"
