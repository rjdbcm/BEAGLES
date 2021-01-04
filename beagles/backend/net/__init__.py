"""Top-level module for machine-learning backend"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import csv
import math
from functools import partial
from multiprocessing.pool import ThreadPool
import av
import numpy as np
import tensorflow as tf
from beagles.base import GradientNaN, Timer
from beagles.io.flags import SharedFlagIO
from beagles.backend.darknet import Darknet
from beagles.backend.net.framework import Framework
from beagles.backend.net.hparams import cyclic_learning_rate as clr


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
            loss = self.loss(x, *y)
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
        self.info = self.io.log.info
        self.pool = ThreadPool()
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
        self.build_optimizer()
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
        del self.checkpoint
        del self.manager
        tf.keras.backend.clear_session()
        gc.collect()

    def build_optimizer(self):
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        # setup kwargs for trainer
        kwargs = dict()
        if self.flags.clip:
            kwargs.update({'clipnorm': self.flags.clip_norm})

        if self.flags.trainer in MOMENTUM_USERS:
            kwargs.update({MOMENTUM: self.flags.momentum})
        elif self.flags.trainer is NESTEROV:
            kwargs.update({self.flags.trainer: True})
        elif self.flags.trainer is AMSGRAD:
            kwargs.update({AMSGRAD.lower(): True})
        elif self.flags.trainer is ADAM:
            kwargs.update({'epsilon': 1.0})

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
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
            self.info(f"Restored from {self.manager.latest_checkpoint}")
        elif self.flags.load >= 1:
            self.info(f"Restoring from {self.flags.load}")
            try:
                [ckpt] = [i for i in self.manager.checkpoints if i.endswith('-'+str(self.flags.load))]
            except ValueError:
                name, _ = os.path.splitext(os.path.basename(self.flags.model))
                name = f"{name}-{self.flags.load}"
                path = os.path.join(self.flags.backup, name)
                raise FileNotFoundError(f'Checkpoint {path} does not exist in {self.manager.checkpoints}.') from None
            self.checkpoint.restore(ckpt).expect_partial()
        else:
            self.info("Initializing network weights from scratch.")

    def preprocess_batch(self, batch):
        preprocess = lambda inp: self.framework.preprocess(os.path.join(self.flags.imgdir, inp))
        return self.pool.map(preprocess, batch)

    def postprocess_batch(self, batch, net_out):
        postprocess = lambda i, pred: self.framework.postprocess(pred, os.path.join(self.flags.imgdir, batch[i]))
        return self.pool.map(lambda p: postprocess(*p), enumerate(net_out))

    def train(self):
        self._build()
        self._load_checkpoint()
        self.io.io_flags()
        self.info('Building {} train op'.format(self.flags.model))
        n = len(self.annotation_data)
        goal = n * self.flags.epoch
        batch_per_epoch = int(n/self.flags.batch)
        saved_last_time = False
        first = True
        for i, (x_batch, loss_feed) in enumerate(self.framework.shuffle(self.annotation_data, class_weights=self.class_weights)):
            self.io.send_flags()
            if saved_last_time and not first:
                self._destroy()
                self._build(rebuild=True)
                self.checkpoint.restore(self.manager.latest_checkpoint)
                saved_last_time = False
                first = True
            loss = self.net.train_step((x_batch, loss_feed))
            self.io.raise_if(not tf.math.is_finite(loss), GradientNaN,
                             "Gradient could not be followed.")
            step = self.net.step.numpy()
            lr = self.net.optimizer.learning_rate.numpy()
            line = '\tstep: {} loss: {:f} lr: {:.2e} progress: {:.2f}%'
            if not first:
                self.flags.progress = i * self.flags.batch / goal * 100
                self.info(line.format(step, loss, lr, self.flags.progress))
            else:
                self.info(f"Following gradient from step {step}...")
            saving = not bool((i + 1) % (batch_per_epoch * self.flags.save))
            if saving:
                save = self.manager.save()
                self.info(f"Saved checkpoint: {save}")
                saved_last_time = True
            first = False

    def img_path(self, image):
        return os.path.join(self.flags.imgdir, image)

    def _postprocess(self, idx, pred, batch):
        return self.framework.postprocess(pred, self.img_path(batch[idx]))

    def _preprocess(self, inp):
        return self.framework.preprocess(self.img_path(inp))

    def predict(self):
        self._build()
        self._load_checkpoint()
        frmwrk = self.framework
        fl = self.flags
        inputs = [i for i in os.listdir(fl.imgdir) if frmwrk.is_input(self.img_path(i))]
        n = len(inputs)
        self.info(f'Predicting annotations for {n} inputs in {fl.imgdir}')
        self.io.raise_if(not inputs, FileNotFoundError,
                         f'Failed to find any valid images in {fl.imgdir}')
        batch = min(fl.batch, n)
        n_batch = int(math.ceil(n / batch))
        preprocess_times, forwarding_times, postprocess_times = [], [], []
        times = [preprocess_times, forwarding_times, postprocess_times]
        for j in range(n_batch):
            start = j * batch
            stop = min(start + batch, len(inputs))
            this_batch = inputs[start:stop]
            self.io.log.info(f'Preprocessing {batch} inputs...')
            with Timer() as t:
                x = self.pool.map(self._preprocess, this_batch)
            self.io.log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
            preprocess_times += [batch/t.elapsed_secs]
            self.io.log.info(f'Forwarding {batch} inputs...')
            with Timer() as t:
                x = [self.net(np.expand_dims(i, 0)) for i in x]
            self.io.log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
            forwarding_times += [batch/t.elapsed_secs]
            self.io.log.info(f'Postprocessing {batch} inputs...')
            with Timer() as t:
                self.pool.map(lambda p: self._postprocess(*p, this_batch), enumerate(x))
            self.io.log.info(f'Done! ({batch/t.elapsed_secs:.2f} inputs/s)')
            postprocess_times += [batch/t.elapsed_secs]
            self.flags.progress += (batch / len(inputs)) * 100.0
            self.io.io_flags()
        self.info(f'Mean preprocess rate:  {np.average(times[0]):.2f} inputs/sec')
        self.info(f'Mean forwarding rate:  {np.average(times[1]):.2f} inputs/sec')
        self.info(f'Mean postprocess rate: {np.average(times[2]):.2f} inputs/sec')
        self.info(f'Mean overall rate:     {np.average(np.concatenate(times)):.2f} inputs/sec')

    def annotate(self, update=10):
        self._build()
        self._load_checkpoint()
        frmwrk = self.framework
        flags = self.flags
        for target in flags.video:
            # noinspection PyUnresolvedReferences
            container = av.open(target)
            total_frames = container.streams.video[0].frames
            container.streams.video[0].thread_type = 'AUTO'
            annotation_file = f'{os.path.splitext(target)[0]}_annotations.csv'
            if os.path.exists(annotation_file):
                self.io.log.info("Overwriting existing annotations")
                os.remove(annotation_file)
            self.io.log.info(f'Annotating {target}')
            file = open(annotation_file, mode='a')
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=0)
            for i, frame in enumerate(container.decode(video=0)):
                if i % update == 0:
                    frame_prog = (update / total_frames) * (1 / len(flags.video))
                    flags.progress += round(100 * frame_prog, 0)
                    self.io.io_flags()
                timestamp = float(frame.pts * frame.time_base)
                frame = frame.to_rgb().to_ndarray()
                h, w, _ = frame.shape
                im = frmwrk.resize_input(frame)
                this_inp = np.expand_dims(im, 0)
                boxes = frmwrk.find(self.net(this_inp))
                pred = [frmwrk.process(b, h, w, flags.threshold) for b in boxes]
                [file_writer.writerow([timestamp, *res]) for res in filter(None, pred)]
            file.close()
            flags.progress = 0.0
            self.io.io_flags()
            self.io.log.info("Done!")
