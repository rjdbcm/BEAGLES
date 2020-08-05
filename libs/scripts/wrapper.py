import os
import sys
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
    from libs.utils.flags import Flags, FlagIO  # Move to the toplevel folder since flag paths are relative to BEAGLES.py
except ModuleNotFoundError:
    sys.path.append(EXEC_PATH)
finally:
    from libs.net.build import TFNet
    from libs.utils.flags import Flags, FlagIO
    os.chdir(EXEC_PATH)


class DarkWrapper(FlagIO):
    def __init__(self):
        FlagIO.__init__(self, subprogram=True)
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
            parser.add_argument('--fbf', default=Flags().fbf,
                                help='generate frame-by-frame annotation')
            parser.add_argument('--saveVideo', default=Flags().saveVideo,
                                help='filename of video output')
            parser.add_argument('--json', default=Flags().json,
                                action='store_true',
                                help='output bounding box information in .json')
            parser.add_argument('--imgdir', default=Flags().imgdir,
                                metavar='',
                                help='path to testing directory with images')
            parser.add_argument('--binary', default=Flags().binary,
                                metavar='',
                                help='path to .weights directory')
            parser.add_argument('--config', default=Flags().config,
                                metavar='',
                                help='path to .cfg directory')
            parser.add_argument('--dataset', default=Flags().dataset,
                                metavar='',
                                help='path to dataset directory')
            parser.add_argument('--backup', default=Flags().backup,
                                metavar='',
                                help='path to checkpoint directory')
            parser.add_argument('--labels', default=Flags().labels,
                                metavar='',
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
            parser.add_argument('--keep', default=Flags().keep,
                                metavar='N',
                                help='number of most recent training results to save')
            parser.add_argument('--batch', default=Flags().batch,
                                metavar='N', help='batch size')
            parser.add_argument('--epoch', default=Flags().epoch,
                                type=int, metavar='N', help='number of epochs')
            parser.add_argument('--save', default=Flags().save,
                                metavar='N',
                                help='save a checkpoint ever N training examples')
            parser.add_argument('--gpu', default=Flags().gpu,
                                metavar='[0 .. 1.0]',
                                help='amount of GPU to use')
            parser.add_argument('--gpuName', default=Flags().gpuName,
                                metavar='/gpu:N', help='GPU device name')
            parser.add_argument('-l', '--load', default=Flags().load,
                                metavar='',
                                help='filename of weights or checkpoint to load')
            parser.add_argument('-m', '--model', default=Flags().model,
                                metavar='', help='filename of model to use')
            parser.add_argument('--threshold', default=Flags().threshold,
                                metavar='[0.01 .. 0.99]',
                                help='threshold of confidence to record an annotation hit')
            parser.add_argument('--clip', default=Flags().clip,
                                help="clip if gradient explodes")
            parser.add_argument('--lr', default=Flags().lr,
                                metavar='N', help='learning rate')
            parser.add_argument('--max_lr', default=Flags().max_lr,
                                metavar='N', help='max learning rate')
            parser.add_argument('-v', '--verbalise', default=Flags().verbalise,
                                action='store_true',
                                help='show graph structure while building')
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
        if self.flags.train:
            TFNet(self.flags).train()
        elif self.flags.fbf != '':
            TFNet(self.flags).annotate()
        else:
            TFNet(self.flags).predict()
        self.done()

    def done(self):
        self.read_flags()
        self.flags.progress = 100
        self.flags.done = True
        self.io_flags()
        exit(0)


if __name__ == '__main__':
    DarkWrapper()
