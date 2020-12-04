"""Top-level module for machine-learning backend"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import csv
import math
from functools import partial
from multiprocessing.pool import ThreadPool
import cv2
import numpy as np
import tensorflow as tf
from beagles.base import GradientNaN, Timer
from beagles.io.flags import SharedFlagIO
from beagles.backend.darknet import Darknet
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
        self.io = SharedFlagIO(subprogram=True)
        self.flags = self.io.read_flags() if self.io.read_flags() else self.io.flags
        self.step = step
        self.first = True

    def _call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x = self._call(x)
            loss = self.loss(x, **y)
            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
        if not self.first:
            # just remembering weights on the first train step
            self.step.assign_add(1)
        self.first = False
        return loss

    def call(self, x, *args):
        x = self._call(x)
        return x

    def get_config(self):
        pass


class NetBuilder(tf.Module):
    """Initializes with flags that build a Darknet or with a prebuilt Darknet.
    Constructs the actual :obj:`Net` object upon being called.

    """
    def __init__(self, flags, darknet=None):
        super(NetBuilder, self).__init__(name=self.__class__.__name__)
        flags = flags if flags else None
        self.io = SharedFlagIO(subprogram=True)
        self.flags = self.io.read_flags() if self.io.read_flags() is not None else flags
        self.darknet = Darknet(flags) if darknet is None else darknet
        self.num_layer = self.ntrain = len(self.darknet.layers) or 0
        self.meta = self.darknet.meta

    def _build(self, rebuild=False):
        if not rebuild:
            self.global_step = tf.Variable(0, trainable=False)
            self.framework = Framework.create(self.darknet.meta, self.flags)
            self.annotation_data, self.class_weights = self.framework.parse()
        self.io.log.info('Compiling Net...')
        self.optimizer = self._build_optimizer()
        self.layers = self.darknet.compile()
        self.net = Net(self.layers, self.global_step, dtype=tf.float32)
        self.net.compile(loss=self.framework.loss, optimizer=self.optimizer)
        ckpt_kwargs = {'net': self.net, 'optimizer': self.optimizer}
        self.checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
        args = [self.checkpoint, self.flags.backup, self.flags.keep]
        self.manager = tf.train.CheckpointManager(*args, checkpoint_name=f"{self.meta['name']}")

    def _destroy(self):
        del self.net
        del self.optimizer
        del self.layers
        tf.keras.backend.clear_session()
        gc.collect()

    def _build_optimizer(self):
        # setup kwargs for trainer
        kwargs = dict()
        if self.flags.trainer in MOMENTUM_USERS:
            kwargs.update({MOMENTUM: self.flags.momentum})
        elif self.flags.trainer is NESTEROV:
            kwargs.update({self.flags.trainer: True})
        elif self.flags.trainer is AMSGRAD:
            kwargs.update({AMSGRAD.lower(): True})
        elif self.flags.trainer is ADAM:
            kwargs.update({'epsilon': 1.0})

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
        return TRAINERS[self.flags.trainer](learning_rate=clr(**clr_kwargs), **kwargs)

    def _load_checkpoint(self):
        if self.flags.load < 0:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            self.io.log.info(f"Restored from {self.manager.latest_checkpoint}")
        elif self.flags.load >= 1:
            self.io.log.info(f"Restoring from {self.flags.load}")
            try:
                [ckpt] = [i for i in self.manager.checkpoints if i.endswith(str(self.flags.load))]
            except ValueError:
                name, _ = os.path.splitext(os.path.basename(self.flags.model))
                name = f"{name}-{self.flags.load}"
                path = os.path.join(self.flags.backup, name)
                raise FileNotFoundError(f'Checkpoint {path} does not exist.') from None
            self.checkpoint.restore(ckpt)
        else:
            self.io.log.info("Initializing network weights from scratch.")

    def train(self):
        self._build()
        self._load_checkpoint()
        self.io.io_flags()
        self.io.log.info('Building {} train op'.format(self.flags.model))
        goal = len(self.annotation_data) * self.flags.epoch
        saved_last_time = False
        first = True
        batch_per_epoch = int(len(self.annotation_data) / self.flags.batch)
        for i, (x_batch, loss_feed) in enumerate(self.framework.shuffle(self.annotation_data, self.class_weights)):
            self.io.io_flags()
            if saved_last_time and not first:
                self._destroy()
                self._build(rebuild=True)
                self.checkpoint.restore(self.manager.latest_checkpoint)
                saved_last_time = False
                first = True
            loss = self.net.train_step((x_batch, loss_feed))
            if not tf.math.is_finite(loss):
                self.io.log.info("Gradient could not be followed.")
                try:
                    raise GradientNaN(flags=self.flags)
                except GradientNaN as e:
                    self.flags.error = e.message
                    self.flags.kill = True
                    self.io.send_flags()
                    raise
            step = self.net.step.numpy()
            lr = self.net.optimizer.learning_rate.numpy()
            line = 'step: {} loss: {:f} lr: {:.2e} progress: {:.2f}%'
            if not first:
                self.flags.progress = i * self.flags.batch / goal * 100
                self.io.log.info(line.format(step, loss, lr, self.flags.progress))
            else:
                self.io.log.info(f"Following gradient from step {step}...")
            saving = not bool((i + 1) % (batch_per_epoch * self.flags.save))
            if saving:
                save = self.manager.save()
                self.io.log.info(f"Saved checkpoint: {save}")
                saved_last_time = True
            first = False

    def predict(self):
        self._build()
        self._load_checkpoint()
        img_path = partial(os.path.join, self.flags.imgdir)
        pool = ThreadPool()
        inputs = [i for i in os.listdir(self.flags.imgdir) if self.framework.is_input(i)]
        if not inputs:
            raise FileNotFoundError(f'Failed to find any valid images in {self.flags.imgdir}')
        batch = min(self.flags.batch, len(inputs))
        n_batch = int(math.ceil(len(inputs) / batch))
        preprocess_times = []
        postprocess_times = []
        forwarding_times= []
        for j in range(n_batch):
            start = j * batch
            stop = min(start + batch, len(inputs))
            this_batch = inputs[start:stop]
            self.io.log.info(f'Preprocessing {batch} inputs...')
            with Timer() as t:
                preprocess = lambda inp: self.framework.preprocess(img_path(inp))
                x = pool.map(preprocess, this_batch)
            self.io.log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
            preprocess_times += [batch/t.elapsed_secs]
            self.io.log.info(f'Forwarding {batch} inputs...')
            with Timer() as t:
                x = [np.concatenate(self.net(np.expand_dims(i, 0)), 0) for i in x]
            self.io.log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
            forwarding_times += [batch/t.elapsed_secs]
            self.io.log.info(f'Postprocessing {batch} inputs...')
            with Timer() as t:
                postprocess = lambda i, pred: self.framework.postprocess(pred, img_path(this_batch[i]))
                pool.map(lambda p: postprocess(*p), enumerate(x))
            self.io.log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
            postprocess_times += [batch/t.elapsed_secs]
            self.flags.progress += (batch / len(inputs)) * 100.0
            self.io.io_flags()
        all_times = [*preprocess_times, *forwarding_times, *postprocess_times]
        self.io.log.info(f'Mean preprocess rate:  {np.average(preprocess_times):.2f} inputs/sec')
        self.io.log.info(f'Mean forwarding rate:  {np.average(forwarding_times):.2f} inputs/sec')
        self.io.log.info(f'Mean postprocess rate: {np.average(postprocess_times):.2f} inputs/sec')
        self.io.log.info(f'Mean overall rate:     {np.average(all_times):.2f} inputs/sec')


    def annotate(self, update=10):
        self._build()
        self._load_checkpoint()
        for video in self.flags.video:
            frame_count = 0
            capture = cv2.VideoCapture(video)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            annotation_file = f'{os.path.splitext(video)[0]}_annotations.csv'
            if os.path.exists(annotation_file):
                self.io.log.info("Overwriting existing annotations")
                os.remove(annotation_file)
            self.io.log.info(f'Annotating {video}')
            with open(annotation_file, mode='a') as file:
                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=0)
            for i in range(total_frames):
                frame_count += 1
                if frame_count % update == 0:
                    frame_prog = (update / total_frames) * (1 / len(self.flags.video))
                    self.flags.progress += round(100 * frame_prog, 0)
                    self.io.io_flags()
                ret, frame = capture.read()
                if ret:
                    frame = np.asarray(frame)
                    h, w, _ = frame.shape
                    im = self.framework.resize_input(frame)
                    this_inp = np.expand_dims(im, 0)
                    boxes = self.framework.findboxes(np.concatenate(self.net(this_inp), 0))
                    pred = [self.framework.process_box(b, h, w, self.flags.threshold) for b in boxes]
                    pred = filter(None, pred)
                    time_elapsed = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    for result in pred:
                        file_writer.writerow([time_elapsed, *result])
            capture.release()
            file.close()
            self.flags.progress = 0.0
            self.io.io_flags()
            self.io.log.info("Done!")
