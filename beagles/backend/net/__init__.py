"""Top-level module for machine-learning backend"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import csv
import math
from functools import partial
from multiprocessing.pool import ThreadPool
import cv2
import numpy as np
import tensorflow as tf
from beagles.base import GradientNaN, Timer
from beagles.io import get_logger
from beagles.io.flags import SharedFlagIO
from beagles.backend.darknet import Darknet
from beagles.backend.net.ops import op_create
from beagles.backend.net.framework import Framework
from beagles.backend.net.hyperparameters import cyclic_learning_rate as clr

MOMENTUM = 'momentum'
NESTEROV = 'nesterov'
AMSGRAD = 'AMSGrad'
RMSPROP = 'rmsprop'
ADADELTA = 'adadelta'
ADAGRAD = 'adagrad'
ADAM = 'adam'
FTRL = 'ftrl'
SGD = 'sgd'

MOMENTUM_USERS = [MOMENTUM, RMSPROP, NESTEROV]
TRAINERS = {
    RMSPROP:  tf.keras.optimizers.RMSprop,
    ADADELTA: tf.keras.optimizers.Adadelta,
    ADAGRAD:  tf.keras.optimizers.Adagrad,
    MOMENTUM: tf.keras.optimizers.SGD,
    NESTEROV: tf.keras.optimizers.SGD,
    ADAM:     tf.keras.optimizers.Adam,
    AMSGRAD:  tf.keras.optimizers.Adam,
    FTRL:     tf.keras.optimizers.Ftrl,
    SGD:      tf.keras.optimizers.SGD
}


class Net(tf.keras.Model):
    """A simple model.
    Args:
        layers: list of :obj:`beagles.backend.darknet.darknet.Darknet` layers
        step: scalar holding current step
    """
    def __init__(self, layers: list, step, **kwargs):
        super(Net, self).__init__(**kwargs)
        for i, layer in enumerate(layers):
            setattr(self, '_'.join([layer.lay.type, str(i)]), layer)
        self.step = step
        self.first = True

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            for layer in self.layers:
                x = layer(x)
            loss = self.loss(x, **y)
        if not tf.math.is_finite(loss):
            raise GradientNaN
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def call(self, x, training=False, **loss_feed):
        with tf.GradientTape() as t:
            for layer in self.layers:
                x = layer(x)
            if training:
                loss = self.loss(x, **loss_feed)
                if not tf.math.is_finite(loss):
                    raise GradientNaN
                variables = self.trainable_variables
                gradients = t.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))
                if not self.first:
                    # just remembering weights on the first train step
                    self.step.assign_add(1)
                self.first = False

        return loss if training else x

class NetBuilder(tf.Module):
    """Initializes with flags that build a Darknet or with a prebuilt Darknet.
    Constructs the actual :obj:`Net` object upon being called.

    """
    def __init__(self, flags, darknet=None):
        super(NetBuilder, self).__init__(name=self.__class__.__name__)
        tf.autograph.set_verbosity(0)
        self.io = SharedFlagIO(subprogram=True)
        self.flags = self.io.read_flags() if self.io.read_flags() is not None else flags
        self.io_flags = self.io.io_flags
        self.logger = get_logger()
        self.darknet = Darknet(flags) if darknet is None else darknet
        self.num_layer = self.ntrain = len(self.darknet.layers) or 0
        self.meta = self.darknet.meta

    def __call__(self):
        self.global_step = tf.Variable(0, trainable=False)
        framework = Framework.create(self.darknet.meta, self.flags)
        self.annotation_data, self.class_weights = framework.parse()
        optimizer = self.build_optimizer()
        layers = self.compile_darknet()
        net = Net(layers, self.global_step, dtype=tf.float32)
        ckpt_kwargs = {'net': net, 'optimizer': optimizer}
        self.checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
        name = f"{self.meta['name']}"
        manager = tf.train.CheckpointManager(self.checkpoint, self.flags.backup,
                                             self.flags.keep, checkpoint_name=name)
        # try to load a checkpoint from flags.load
        self.load_checkpoint(manager)
        self.logger.info('Compiling Net...')
        net.compile(loss=framework.loss, optimizer=optimizer)
        return net, framework, manager

    def build_optimizer(self):
        # setup kwargs for trainer
        kwargs = dict()
        if self.flags.trainer in MOMENTUM_USERS:
            kwargs.update({MOMENTUM: self.flags.momentum})
        if self.flags.trainer is NESTEROV:
            kwargs.update({self.flags.trainer: True})
        if self.flags.trainer is AMSGRAD:
            kwargs.update({AMSGRAD.lower(): True})
        if self.flags.clip:
            kwargs.update({'clipnorm': self.flags.clip_norm})
        ssc = self.flags.step_size_coefficient
        step_size = int(ssc * (len(self.annotation_data) // self.flags.batch))
        clr_kwargs = {
            'global_step': self.global_step,
            'mode': self.flags.clr_mode,
            'step_size': step_size,
            'learning_rate': self.flags.lr,
            'max_lr': self.flags.max_lr,
            'name': self.flags.model
        }
        # setup trainer
        return TRAINERS[self.flags.trainer](learning_rate=lambda: clr(**clr_kwargs), **kwargs)

    def compile_darknet(self):
        layers = list()
        roof = self.num_layer - self.ntrain
        prev = None
        for i, layer in enumerate(self.darknet.layers):
            layer = op_create(layer, prev, i, roof)
            layers.append(layer)
            prev = layer
        return layers

    def load_checkpoint(self, manager):
        if isinstance(self.flags.load, str):
            checkpoint = [i for i in manager.checkpoints if self.flags.load in i]
            assert len(checkpoint) == 1
            self.checkpoint.restore(checkpoint)
            self.logger.info(f"Restored from {checkpoint}")
        elif self.flags.load < 0:
            self.checkpoint.restore(manager.latest_checkpoint)
            self.logger.info(f"Restored from {manager.latest_checkpoint}")
        elif self.flags.load >= 1:
            idx = self.flags.load - 1
            self.checkpoint.restore(manager.checkpoints[idx])
            self.logger.info(f"Restored from {manager.checkpoints[idx]}")
        else:
            self.logger.info("Initializing network weights from scratch.")


def train(data, class_weights, flags, net: Net, framework: Framework, manager: tf.train.CheckpointManager):
    log = get_logger()
    io = SharedFlagIO(flags, subprogram=True)
    flags = io.read_flags() if io.read_flags() is not None else flags
    log.info('Building {} train op'.format(flags.model))
    goal = len(data) * flags.epoch
    first = True
    for i, (x_batch, loss_feed) in enumerate(framework.shuffle(data, class_weights)):
        loss = net(x_batch, training=True, **loss_feed)
        step = net.step.numpy()
        lr = net.optimizer.learning_rate.numpy()
        line = 'step: {} loss: {:f} lr: {:.2e} progress: {:.2f}%'
        if not first:
            flags.progress = i * flags.batch / goal * 100
            log.info(line.format(step, loss, lr, flags.progress))
        else:
            log.info(f"Following gradient from step {step}...")
        io.send_flags()
        flags = io.read_flags()
        ckpt = bool(not step % flags.save)
        if ckpt and not first:
            save = manager.save()
            log.info(f"Saved checkpoint: {save}")
        first = False
    if not ckpt:
        save = manager.save()
        log.info(f"Finished training at checkpoint: {save}")

def predict(flags, net: Net, framework: Framework):
    log = get_logger()
    io = SharedFlagIO(flags, subprogram=True)
    pool = ThreadPool()
    flags = io.read_flags() if io.read_flags() is not None else flags
    all_inps = [i for i in os.listdir(flags.imgdir) if framework.is_input(i)]
    if not all_inps:
        raise FileNotFoundError(f'Failed to find any images in {flags.imgdir}')
    batch = min(flags.batch, len(all_inps))
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        start = j * batch
        stop = min(start + batch, len(all_inps))
        this_batch = all_inps[start:stop]
        img_path = partial(os.path.join, flags.imgdir)
        log.info(f'Preprocessing {batch} inputs...')
        with Timer() as t:
            x = pool.map(lambda inp: framework.preprocess(img_path(inp)), this_batch)
        log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
        log.info(f'Forwarding {batch} inputs...')
        with Timer() as t:
            x = [np.concatenate(net(np.expand_dims(i, 0)), 0) for i in x]
        log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
        log.info(f'Postprocessing {batch} inputs...')
        with Timer() as t:
            postprocess = lambda i, pred: framework.postprocess(pred, img_path(this_batch[i]))
            pool.map(lambda p: postprocess(*p), enumerate(x))
        log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')

def annotate(flags, net, framework):
    log = get_logger()
    io = SharedFlagIO(flags, subprogram=True)
    flags = io.read_flags() if io.read_flags() is not None else flags
    for video in flags.video:
        frame_count = 0
        capture = cv2.VideoCapture(video)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        annotation_file = f'{os.path.splitext(video)[0]}_annotations.csv'
        if os.path.exists(annotation_file):
            log.info("Overwriting existing annotations")
            os.remove(annotation_file)
        log.info(f'Annotating {video}')
        with open(annotation_file, mode='a') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            while capture.isOpened():
                frame_count += 1
                if frame_count % 10 == 0:
                    flags.progress = round((100 * frame_count / total_frames), 0)
                    io.io_flags()
                ret, frame = capture.read()
                if ret:
                    frame = np.asarray(frame)
                    h, w, _ = frame.shape
                    im = framework.resize_input(frame)
                    this_inp = np.expand_dims(im, 0)
                    boxes = framework.findboxes(np.concatenate(net(this_inp), 0))
                    pred = [framework.process_box(b, h, w, flags.threshold) for b in boxes]
                    pred = filter(None, pred)
                    time_elapsed = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    [file_writer.writerow([time_elapsed, *result]) for result in pred]
                else:
                    break
                if flags.kill:
                    capture.release()
                    exit(1)
        capture.release()
