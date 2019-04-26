#! /usr/bin/env python

# will fix repetitious self.parser.add_argument() statements at some point with parent/child parsers
# Will add image processing to demo command
# will add thread-based frame buffer for video processing

import os
import argparse
from darkflow.net.build import TFNet

class Flow(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='[dark]flow translates darknet to tensorflow')
        self.parser.add_argument('--train', default=False, action='store_true',
                            help='train a model on annotated data')
        self.parser.add_argument('--savepb', default=False, action='store_true',
                            help='freeze the model to a .pb')
        self.parser.add_argument('--demo', default='', help='demo model on video or webcam')
        self.parser.add_argument('--fbf', default='', help='generate frame-by-frame annotation')
        self.parser.add_argument('--saveVideo', default='out.avi', help='filename of video output')
        self.parser.add_argument('--queue', default=1, help='batch process demo')
        self.parser.add_argument('--json', default=False, action='store_true',
                            help='output bounding box information in .json')
        self.parser.add_argument('--imgdir', default='./sample_img/', metavar='',
                            help='path to testing directory with images')
        self.parser.add_argument('--binary', default='./bin/', metavar='', help='path to .weights directory')
        self.parser.add_argument('--config', default='./cfg/', metavar='', help='path to .cfg directory')
        self.parser.add_argument('--dataset', default='../data/rawframes', metavar='',
                            help='path to dataset directory')
        self.parser.add_argument('--backup', default='./ckpt/', metavar='', help='path to checkpoint directory')
        self.parser.add_argument('--labels', default='../data/predefined_classes.txt', metavar='', help='path to textfile containing labels')
        self.parser.add_argument('--annotation', default='../data/rawframes', metavar='',
                            help='path to the annotation directory')
        self.parser.add_argument('--summary', default='', help='path to Tensorboard summaries directory')
        self.parser.add_argument('--trainer', default='rmsprop', metavar='', help='training algorithm')
        self.parser.add_argument('--momentum', default=0.0, metavar='',
                            help='applicable for rmsprop and momentum optimizers')
        self.parser.add_argument('--keep', default=20, metavar='N', help='number of most recent training results to save')
        self.parser.add_argument('--batch', default=16, metavar='N', help='batch size')
        self.parser.add_argument('--epoch', default=1000, metavar='N', help='number of epochs')
        self.parser.add_argument('--save', default=2000, metavar='N', help='save a checkpoint ever N training examples')
        self.parser.add_argument('--pbLoad', default='', metavar='*.pb', help='name of protobuf file to load')
        self.parser.add_argument('--metaLoad', default='', metavar='',
                            help='path to .meta file generated during --savepb that corresponds to .pb file')
        self.parser.add_argument('--gpu', default=0.0, metavar='[0 .. 1.0]', help='amount of GPU to use')
        self.parser.add_argument('--gpuName', default='/gpu:0', metavar='/gpu:N', help='GPU device name')
        self.parser.add_argument('-l', '--load', default=-1, metavar='', help='filename of weights or checkpoint to load')
        self.parser.add_argument('-m', '--model', default='cfg/tiny-yolo-4c.cfg', metavar='', help='filename of model to use')
        self.parser.add_argument('--threshold', default=-0.1, metavar='[0.01 .. 0.99]',
                            help='threshold of confidence to record an annotation hit')
        self.parser.add_argument('--lr', default=1e-5, metavar='N', help='learning rate')
        self.parser.add_argument('-v', '--verbalise', default=False, action='store_true',
                            help='show graph structure while building')

        FLAGS = self.parser.parse_args()

        def _get_dir(dirs):
            for d in dirs:
                this = os.path.abspath(os.path.join(os.path.curdir, d))
                if not os.path.exists(this):
                    os.makedirs(this)

        requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir, 'out')]
        if FLAGS.summary:
            requiredDirectories.append(FLAGS.summary)
        _get_dir(requiredDirectories)

        if FLAGS.gpu > 1.0 or FLAGS.gpu < 0:
            raise Exception('--gpu should be a number between 0 and 1')

        try: # TODO: add constraints on FLAGS.gpu range and except: ValueError also add individual error catching for numbers
            FLAGS.save = int(FLAGS.save)
            FLAGS.epoch = int(FLAGS.epoch)
            FLAGS.batch = int(FLAGS.batch)
            FLAGS.threshold = float(FLAGS.threshold)
            FLAGS.gpu = float(FLAGS.gpu)
            FLAGS.lr = float(FLAGS.lr)
        except ValueError:
            print('You should try using numbers instead.')

        try:
            FLAGS.load = int(FLAGS.load)
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
    Flow()
