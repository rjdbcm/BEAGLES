from unittest import TestCase
from libs.utils.flags import FlagIO, Flags


class TestFlagIO(TestCase):
    def testRAMDisk(self):
        self.io = FlagIO()
        self.io.flags = Flags()
        self.io.init_ramdisk()
        self.io.io_flags()
        self.io.cleanup_ramdisk()