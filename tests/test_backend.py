from unittest import TestCase
from libs.io.flags import FlagIO
from libs.utils.flags import Flags
from libs.utils.errors import GradientNaN
from libs.cythonUtils import iou_c


class TestBackend(TestCase, FlagIO):
    def setUp(self):
        FlagIO.__init__(self)
        self.flags = Flags()
        self.send_flags()
        self.flags.model = 'data/cfg/tiny-yolov2.cfg'

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
