import os
from libs.net.build import TFNet
from libs.utils.flags import Flags, FlagIO  # Move to the toplevel folder since flag paths are relative to slgrSuite.py
import sys

class DarkWrapper(FlagIO):
    def __init__(self):
        FlagIO.__init__(self, subprogram=True)
        print(os.getcwd())
        self.flags = self.read_flags()
        self.flags.started = True
        self.io_flags()
        FLAGS = self.flags
        if FLAGS.train:
            TFNet(FLAGS).train()
            self.read_flags()
            self.flags.progress = 100
            self.flags.done = True
            self.io_flags()
            exit(0)
        elif FLAGS.savepb:
            print('[INFO] Freezing graph of {} at {} to a protobuf file...'.format(FLAGS.model, FLAGS.load))
            TFNet(FLAGS).savepb()
            self.read_flags()
            self.flags.progress = 100
            self.flags.done = True
            self.io_flags()
            exit('[INFO] Done')
        elif FLAGS.demo != '':
            tfnet.camera()
        elif FLAGS.fbf != '':
            tfnet.annotate()
        else:
            tfnet.predict()


if __name__ == '__main__':
    DarkWrapper()
