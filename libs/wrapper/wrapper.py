import os
from libs.net.build import TFNet
from libs.utils.flags import Flags, FlagIO
os.chdir("../../")  # Move to the toplevel folder since flag paths are relative to slgrSuite.py


class DarkWrapper(FlagIO):
    def __init__(self):
        FlagIO.__init__(self, subprogram=True)
        self.flags = Flags()
        self.io_flags()

        def _get_dir(dirs):
            for d in dirs:
                this = os.path.abspath(os.path.join(os.path.curdir, d))
                if not os.path.exists(this):
                    os.makedirs(this)
        FLAGS = self.flags
        requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir, 'out')]
        if FLAGS.summary:
            requiredDirectories.append(FLAGS.summary)
        _get_dir(requiredDirectories)

        try:
            FLAGS.load = int(FLAGS.load)
            FLAGS.threshold = float(FLAGS.threshold)
        except:
            pass  # Non-integer passed as filename using bare except

        tfnet = TFNet(FLAGS)

        if FLAGS.train:
            tfnet.train()
            exit('[INFO] Training finished, exit.')
        elif FLAGS.savepb:
            print('[INFO] Freezing graph of {} at {} to a protobuf file...'.format(FLAGS.model, FLAGS.load))
            tfnet.savepb()
            exit('[INFO] Done')
        elif FLAGS.demo != '':
            tfnet.camera()
        elif FLAGS.fbf != '':
            tfnet.annotate()
        else:
            tfnet.predict()


if __name__ == '__main__':
    DarkWrapper()
