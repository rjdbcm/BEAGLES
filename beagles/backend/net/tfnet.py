import os
import time
import tensorflow as tf
from beagles.backend.net.ops import op_create, identity
from beagles.backend.net.ops.baseop import HEADER, LINE
from beagles.backend.net.framework import Framework
from beagles.backend.darknet.darknet import Darknet
from beagles.backend.io.loader import Loader
from beagles.base import VariableIsNone
from beagles.io.logs import get_logger
from beagles.io.flags import SharedFlagIO
from beagles.backend.net.hyperparameters.cyclic_learning_rate import cyclic_learning_rate as clr


old_graph_msg = 'Resolving old graph def {} (no guarantee)'
MOMENTUM = 'momentum'
NESTEROV = 'nesterov'
AMSGRAD = 'AMSGrad'
TRAINERS = dict({
    'rmsprop': tf.keras.optimizers.RMSprop,
    'adadelta': tf.keras.optimizers.Adadelta,
    'adagrad': tf.keras.optimizers.Adagrad,
    MOMENTUM: tf.keras.optimizers.SGD,
    NESTEROV: tf.keras.optimizers.SGD,
    'adam': tf.keras.optimizers.Adam,
    AMSGRAD: tf.keras.optimizers.Adam,
    'ftrl': tf.keras.optimizers.Ftrl,
    'sgd': tf.keras.optimizers.SGD
})
MOMENTUM_USERS = [MOMENTUM, 'rmsprop', NESTEROV]

class TFNet:

    # Interface Methods:
    def __init__(self, flags, darknet=None):
        self.io = SharedFlagIO(subprogram=True)
        # disable eager mode for TF1-dependent code
        tf.compat.v1.disable_eager_execution()
        self.flags = self.io.read_flags() if self.io.read_flags() is not None else flags
        self.io_flags = self.io.io_flags
        self.logger = get_logger()
        self.ntrain = 0
        darknet = Darknet(flags) if darknet is None else darknet
        self.ntrain = len(darknet.layers)
        self.darknet = darknet
        self.num_layer = len(darknet.layers)
        self.framework = Framework.create(darknet.meta, flags)
        self.annotation_data = self.framework.parse()
        self.meta = darknet.meta
        self.graph = tf.Graph()
        device_name = flags.gpu_name if flags.gpu > 0.0 else None
        start = time.time()
        with tf.device(device_name):
            with self.graph.as_default():
                self.build_forward()
                self.setup_meta_ops()
        self.logger.info('Finished in {}s'.format(time.time() - start))

    def raise_error(self, error: Exception, traceback=None):
        form = "{}\nOriginal Tensorflow Error: {}"
        try:
            raise error
        except Exception as e:
            if traceback:
                oe = traceback.message
                self.flags.error = form.format(str(e), oe)
            else:
                self.flags.error = str(e)
            self.logger.error(str(e))
            self.io.send_flags()
            raise

    def build_forward(self):
        # Placeholders
        inp_size = self.meta['inp_size']
        self.inp = tf.keras.layers.Input(dtype=tf.float32, shape=tuple(inp_size), name='input')
        self.feed = dict()  # other placeholders

        # Build the forward pass
        state = identity(self.inp)
        roof = self.num_layer - self.ntrain
        self.logger.info(LINE)
        self.logger.info(HEADER)
        self.logger.info(LINE)
        for i, layer in enumerate(self.darknet.layers):
            scope = '{}-{}'.format(str(i), layer.type)
            args = [layer, state, i, roof, self.feed]
            state = op_create(*args)
            mess = state.verbalise()
            msg = mess if mess else LINE
            self.logger.info(msg)

        self.top = state
        self.out = tf.identity(state.out, name='output')

    def setup_meta_ops(self):
        tf.config.set_soft_device_placement(False)
        tf.debugging.set_log_device_placement(False)
        utility = min(self.flags.gpu, 1.)
        if utility > 0.0:
            tf.config.set_soft_device_placement(True)
        else:
            self.logger.info('Running entirely on CPU')

        if self.flags.train:
            self.build_train_op()

        if self.flags.summary:
            self.summary_op = tf.compat.v1.summary.merge_all()
            self.writer = tf.compat.v1.summary.FileWriter(self.flags.summary + self.flags.project_name)

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if not self.ntrain:
            return

        try:
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            if self.flags.load != 0:
                self.load_from_ckpt()

        except tf.errors.NotFoundError as e:
            self.flags.error = str(e.message)
            self.send_flags()
            raise

        if self.flags.summary:
            self.writer.add_graph(self.sess.graph)

    def load_from_ckpt(self):
        if self.flags.load < 0:  # load lastest ckpt

            with open(os.path.join(self.flags.backup, 'checkpoint'), 'r') as f:
                last = f.readlines()[-1].strip()
                load_point = last.split(' ')[1]
                load_point = load_point.split('"')[1]
                print(load_point)
                load_point = load_point.split('-')[-1]
                self.flags.load = int(load_point)

        load_point = os.path.join(self.flags.backup, self.meta['name'])
        load_point = '{}-{}'.format(load_point, self.flags.load)
        self.logger.info('Loading from {}'.format(load_point))
        try:
            self.saver.restore(self.sess, load_point)
        except ValueError:
            self.load_old_graph(load_point)

    def load_old_graph(self, ckpt):
        ckpt_loader = Loader.create(ckpt)
        self.logger.info(old_graph_msg.format(ckpt))

        for var in tf.compat.v1.global_variables():
            name = var.name.split(':')[0]
            args = [name, var.get_shape()]
            val = ckpt_loader(*args)
            if val is None:
                self.raise_error(VariableIsNone(var))
            shp = val.shape
            plh = tf.compat.v1.placeholder(tf.float32, shp)
            op = tf.compat.v1.assign(var, plh)
            self.sess.run(op, {plh: val})

    def build_train_op(self):
        self.framework.loss(self.out)
        self.logger.info('Building {} train op'.format(self.meta['model']))
        self.global_step = tf.Variable(0, trainable=False)
        # setup kwargs for trainer
        kwargs = dict()
        if self.flags.trainer in ['momentum', 'rmsprop', 'nesterov']:
            kwargs.update({'momentum': self.flags.momentum})
        if self.flags.trainer == 'nesterov':
            kwargs.update({self.flags.trainer: True})
        if self.flags.trainer == 'AMSGrad':
            kwargs.update({self.flags.trainer.lower(): True})
        if self.flags.clip:
            kwargs.update({'clipnorm': self.flags.clip_norm})

        # setup cyclic_learning_rate args
        ssc = self.flags.step_size_coefficient
        step_size = int(ssc * (len(self.annotation_data) // self.flags.batch))
        clr_kwargs = {
            'global_step':   self.global_step,
            'mode':          self.flags.clr_mode,
            'step_size':     step_size,
            'learning_rate': self.flags.lr,
            'max_lr':        self.flags.max_lr,
            'name': 'learning-rate'
        }

        # setup trainer
        self.optimizer = TRAINERS[self.flags.trainer](clr(**clr_kwargs), **kwargs)

        # setup gradients for all globals except the global_step
        vars = tf.compat.v1.global_variables()[:-1] #
        grads = self.optimizer.get_gradients(self.framework.loss, vars)
        self.train_op = self.optimizer.apply_gradients(zip(grads, vars))


class Net(tf.keras.Model):
    """A simple linear model."""
    def __init__(self, layers: list, **kwargs):
        super(Net, self).__init__(**kwargs)
        for i, layer in enumerate(layers):
            setattr(self, '_'.join([layer.lay.type, str(i)]), layer)

    def call(self, x, training=False, **loss_feed):
        with tf.GradientTape() as t:
            for layer in self.layers:
                x = layer(x)
            loss = self.loss(x, **loss_feed)
            if training: # update gradients
                variables = self.trainable_variables
                gradients = t.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

class NetBuilder(tf.Module):
    """Initializes a net builder with flags that builds a :obj:`Net` upon being called"""
    def __init__(self, flags, darknet=None):
        super(NetBuilder, self).__init__(name=self.__class__.__name__)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
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
        self.annotation_data = framework.parse()
        optimizer = self.build_optimizer()
        layers = self.compile_darknet()
        net = Net(layers, dtype=tf.float32)
        ckpt_kwargs = {
            'step': self.global_step,
            'net': net,
            'optimizer': optimizer,
        }
        self.checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
        name = f"{self.meta['name']}"
        manager = tf.train.CheckpointManager(self.checkpoint, self.flags.backup,
                                             self.flags.keep,
                                             checkpoint_name=name)

        # try to load a checkpoint from flags.load
        self.load_checkpoint(manager)

        if self.flags.train:
            self.train(net, framework, manager, optimizer)

    def train(self, net: Net, framework, manager: tf.train.CheckpointManager,
              optimizer: tf.keras.optimizers.Optimizer):
        self.logger.info('Building {} train op'.format(self.meta['model']))
        goal = len(self.annotation_data) * self.flags.epoch
        net.compile(loss=framework.lossv2, optimizer=optimizer)
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

        batches = framework.shuffle(self.annotation_data)
        for i, (x_batch, loss_feed) in enumerate(batches):
            self.global_step.assign_add(1)
            loss = net(x_batch, training=True, **loss_feed)
            count = i * self.flags.batch
            self.flags.progress = count / goal * 100
            ckpt = self.checkpoint.step.numpy()
            lr = net.optimizer.learning_rate.numpy()
            line = 'step: {} loss: {} lr: {} progress: {}'
            self.logger.info(line.format(ckpt, loss, lr, self.flags.progress))
            # cycle learning rate
            net.optimizer.learning_rate = clr(**clr_kwargs)
            ckpt = i % 10
            if not ckpt:
                manager.save()
        if ckpt:
            manager.save()

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

        # setup trainer
        return TRAINERS[self.flags.trainer](self.flags.lr, **kwargs)

    def compile_darknet(self):
        layers = list()
        roof = self.num_layer - self.ntrain
        for i, layer in enumerate(self.darknet.layers):
            layer = op_create(layer, i, roof)
            layers.append(layer)
        return layers

    def load_checkpoint(self, manager):
        if isinstance(self.flags.load, str):
            checkpoint = [i for i in manager.checkpoints if self.flags.load in i]
            assert len(checkpoint) == 1
            self.checkpoint.restore(checkpoint)
            print(f"Restored from {checkpoint}")
        elif self.flags.load < 0:
            self.checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from {manager.latest_checkpoint}")
        elif self.flags.load > 1:
            idx = self.flags.load - 1
            self.checkpoint.restore(manager.checkpoints[idx])
            print(f"Restored from {manager.checkpoints[idx]}")
        else:
            print("Initializing network weights from scratch.")