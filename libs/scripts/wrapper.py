import os
import sys
import numpy as np
import time
from signal import signal, SIGTERM
import argparse
try:
    argv = sys.argv[1]
except IndexError:
    argv = []
if argv:
    EXEC_PATH = os.path.abspath("../../")
else:
    EXEC_PATH = os.getcwd()
try:
    from libs.net.build import TFNet
    from libs.utils.flags import Flags, FlagIO
    # Move to the toplevel folder since flag paths are relative to slgrSuite.py
except ModuleNotFoundError:
    sys.path.append(EXEC_PATH)
finally:
    from libs.net.build import TFNet
    from libs.utils.flags import Flags, FlagIO
    os.chdir(EXEC_PATH)


class DarkWrapper(FlagIO):
    """This wrapper can be used standalone as a CLI or as part of SLGR-Suite"""
    def __init__(self):
        FlagIO.__init__(self, subprogram=True)
        signal(SIGTERM, self.kill)
        try:
            argv = sys.argv[1]
        except IndexError:
            argv = False
        if argv:
            parser = argparse.ArgumentParser(
                description='[dark]flow translates darknet to tensorflow')
            parser.add_argument('--train', default=Flags().train,
                                action='store_true',
                                help='train a model on annotated data')
            parser.add_argument('--savepb', default=Flags().savepb,
                                action='store_true',
                                help='freeze the model to a .pb')
            parser.add_argument('--demo', default=Flags().demo,
                                help='demo model on video or webcam')
            parser.add_argument('--fbf', default=Flags().fbf,
                                help='generate frame-by-frame annotation')
            parser.add_argument('--saveVideo', metavar='',
                                default=Flags().saveVideo,
                                help='filename of video output')
            parser.add_argument('--json', default=Flags().json,
                                action='store_true',
                                help='output bounding box information in .json')
            parser.add_argument('--imgdir', default=Flags().imgdir, metavar='',
                                help='path to testing directory with images')
            parser.add_argument('--binary', default=Flags().binary, metavar='',
                                help='path to .weights directory')
            parser.add_argument('--config', default=Flags().config, metavar='',
                                help='path to .cfg directory')
            parser.add_argument('--dataset', default=Flags().dataset,
                                metavar='',
                                help='path to dataset directory')
            parser.add_argument('--backup', default=Flags().backup, metavar='',
                                help='path to checkpoint directory')
            parser.add_argument('--labels', default=Flags().labels, metavar='',
                                help='path to textfile containing labels')
            parser.add_argument('--annotation', default=Flags().annotation,
                                metavar='',
                                help='path to the annotation directory')
            parser.add_argument('--summary', default=Flags().summary,
                                help='path to Tensorboard summaries directory')
            parser.add_argument('--log', default=Flags().log,
                                help='path to log directory')
            parser.add_argument('--trainer', default=Flags().trainer,
                                metavar='', help='training algorithm')
            parser.add_argument('--momentum', default=Flags().momentum,
                                metavar='',
                                help='applicable for rmsprop and momentum optimizers')
            parser.add_argument('--keep', default=Flags().keep, metavar='N',
                                help='number of recent training results to save')
            parser.add_argument('--batch', default=Flags().batch, metavar='N',
                                type=int, help='batch size')
            parser.add_argument('--epoch', default=Flags().epoch, type=int,
                                metavar='N', help='number of epochs')
            parser.add_argument('--save', default=Flags().save, metavar='N',
                                help='save a checkpoint ever N training examples')
            parser.add_argument('--pbLoad', default=Flags().pbLoad, metavar='*.pb',
                                help='name of protobuf file to load')
            parser.add_argument('--metaLoad', default=Flags().metaLoad,
                                metavar='',
                                help='path to .meta file corresponding to .pb'
                                     ' file')
            parser.add_argument('--gpu', default=Flags().gpu,
                                metavar='[0 .. 1.0]',
                                help='amount of GPU to use')
            parser.add_argument('--gpuName', default=Flags().gpuName,
                                metavar='/gpu:N', help='GPU device name')
            parser.add_argument('-l', '--load', default=Flags().load,
                                metavar='', type=int,
                                help='filename of checkpoint to load')
            parser.add_argument('-m', '--model', default=Flags().model,
                                metavar='', help='filename of model to use')
            parser.add_argument('--threshold', default=Flags().threshold,
                                type=float, choices=np.arange(0.01, 1.0, 0.01),
                                metavar='[0.01 .. 0.99]',
                                help='threshold of confidence')
            parser.add_argument('--clip', default=Flags().clip,
                                help="clip if gradient explodes")
            parser.add_argument('--lr', default=Flags().lr, metavar='N',
                                help='learning rate')
            parser.add_argument('-v', '--verbalise', default=Flags().verbalise,
                                action='store_true',
                                help='show graph structure while building')
            parser.add_argument('--timeout', default=Flags().timeout,
                                metavar="SECONDS",
                                help='capture record time')
            parser.add_argument('--cli', default=True, help=argparse.SUPPRESS)
            parser.add_argument('--kill', default=Flags().kill,
                                help=argparse.SUPPRESS)
            parser.add_argument('--done', default=Flags().done,
                                help=argparse.SUPPRESS)
            parser.add_argument('--started', default=Flags().started,
                                help=argparse.SUPPRESS)
            self.flags = parser.parse_args()
            self.send_flags()
        self.flags = self.read_flags()
        self.flags.started = True
        self.io_flags()
        try:
            if self.flags.train:
                TFNet(self.flags).train()
            elif self.flags.savepb:
                TFNet(self.flags).savepb()
            elif self.flags.demo != '':
                TFNet(self.flags).camera()
            elif self.flags.fbf != '':
                TFNet(self.flags).annotate()
            else:
                TFNet(self.flags).predict()
            self.done()
        except KeyboardInterrupt:
            self.cleanup_ramdisk()
            self.logger.error("Keyboard Interrupt")
            if os.stat(self.logfile.baseFilename).st_size > 0:
                self.logfile.doRollover()
            if os.stat(self.tf_logfile.baseFilename).st_size > 0:
                self.tf_logfile.doRollover()
            raise

    def kill(self, sig, frame):
        self.logger.info("SIGTERM caught: exiting")
        exit(0)

    def done(self):
        self.read_flags()
        self.flags.progress = 100
        self.flags.done = True
        self.logger.info("Operation complete: exiting")
        self.io_flags()
        self.cleanup_ramdisk()
        if os.stat(self.logfile.baseFilename).st_size > 0:
            self.logfile.doRollover()
        if os.stat(self.tf_logfile.baseFilename).st_size > 0:
            self.tf_logfile.doRollover()
        sys.exit(0)


if __name__ == '__main__':
    DarkWrapper()
