import os
import argparse
from libs.net.build import TFNet
from libs.utils.flags import Flags, FlagIO  # Move to the toplevel folder since flag paths are relative to slgrSuite.py
import sys

class DarkWrapper(FlagIO):
    def __init__(self):
        FlagIO.__init__(self, subprogram=True)
        if sys.argv[1]:
            os.chdir("../../")
            FLAGS = Flags()
            parser = argparse.ArgumentParser(
                description='[dark]flow translates darknet to tensorflow')
            parser.add_argument('--train', default=FLAGS.train, action='store_true',
                                help='train a model on annotated data')
            parser.add_argument('--savepb', default=FLAGS.savepb, action='store_true',
                                help='freeze the model to a .pb')
            parser.add_argument('--demo', default=FLAGS.demo, help='demo model on video or webcam')
            parser.add_argument('--fbf', default=FLAGS.fbf, help='generate frame-by-frame annotation')
            parser.add_argument('--saveVideo', default=FLAGS.saveVideo, help='filename of video output')
            parser.add_argument('--queue', default=FLAGS.queue, help='batch process demo')
            parser.add_argument('--json', default=FLAGS.json, action='store_true',
                                help='output bounding box information in .json')
            parser.add_argument('--imgdir', default=FLAGS.imgdir, metavar='',
                                help='path to testing directory with images')
            parser.add_argument('--binary', default=FLAGS.binary, metavar='', help='path to .weights directory')
            parser.add_argument('--config', default=FLAGS.config, metavar='', help='path to .cfg directory')
            parser.add_argument('--dataset', default=FLAGS.dataset, metavar='',
                                help='path to dataset directory')
            parser.add_argument('--backup', default=FLAGS.backup, metavar='', help='path to checkpoint directory')
            parser.add_argument('--labels', default=FLAGS.labels, metavar='', help='path to textfile containing labels')
            parser.add_argument('--annotation', default=FLAGS.annotation, metavar='',
                                help='path to the annotation directory')
            parser.add_argument('--summary', default=FLAGS.summary, help='path to Tensorboard summaries directory')
            parser.add_argument('--log', default=FLAGS.log, help='path to log directory')
            parser.add_argument('--trainer', default=FLAGS.trainer, metavar='', help='training algorithm')
            parser.add_argument('--momentum', default=FLAGS.momentum, metavar='',
                                help='applicable for rmsprop and momentum optimizers')
            parser.add_argument('--keep', default=FLAGS.keep, metavar='N',
                                help='number of most recent training results to save')
            parser.add_argument('--batch', default=FLAGS.batch, metavar='N', help='batch size')
            parser.add_argument('--epoch', default=FLAGS.epoch, metavar='N', help='number of epochs')
            parser.add_argument('--save', default=FLAGS.save, metavar='N',
                                help='save a checkpoint ever N training examples')
            parser.add_argument('--pbLoad', default=FLAGS.pbLoad, metavar='*.pb', help='name of protobuf file to load')
            parser.add_argument('--metaLoad', default=FLAGS.metaLoad, metavar='',
                                help='path to .meta file generated during --savepb that corresponds to .pb file')
            parser.add_argument('--gpu', default=FLAGS.gpu, metavar='[0 .. 1.0]', help='amount of GPU to use')
            parser.add_argument('--gpuName', default=FLAGS.gpuName, metavar='/gpu:N', help='GPU device name')
            parser.add_argument('-l', '--load', default=FLAGS.load, metavar='',
                                help='filename of weights or checkpoint to load')
            parser.add_argument('-m', '--model', default=FLAGS.model, metavar='', help='filename of model to use')
            parser.add_argument('--threshold', default=FLAGS.threshold, metavar='[0.01 .. 0.99]',
                                help='threshold of confidence to record an annotation hit')
            parser.add_argument('--clip', default=FLAGS.clip, help="clip if gradient explodes")
            parser.add_argument('--lr', default=FLAGS.lr, metavar='N', help='learning rate')
            parser.add_argument('-v', '--verbalise', default=FLAGS.verbalise, action='store_true',
                                help='show graph structure while building')
            parser.add_argument('--kill', default=FLAGS.kill, help=argparse.SUPPRESS)
            parser.add_argument('--done', default=FLAGS.done, help=argparse.SUPPRESS)
            parser.add_argument('--started', default=FLAGS.started, help=argparse.SUPPRESS)
            self.flags = parser.parse_args()
            self.send_flags()
        self.flags = self.read_flags()
        self.flags.started = True
        self.io_flags()
        FLAGS = self.flags
        if FLAGS.train:
            TFNet(FLAGS).train()
        elif FLAGS.savepb:
            TFNet(FLAGS).savepb()
        elif FLAGS.demo != '':
            TFNet(FLAGS).camera()
        elif FLAGS.fbf != '':
            TFNet(FLAGS).annotate()
        else:
            TFNet(FLAGS).predict()
        self.done()

    def done(self):
        self.read_flags()
        self.flags.progress = 100
        self.flags.done = True
        self.io_flags()
        exit(0)

if __name__ == '__main__':
    DarkWrapper()
