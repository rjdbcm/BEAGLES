import os
import time
import tensorflow as tf
from beagles.backend.net.ops import op_create, identity
from beagles.backend.net.ops.baseop import HEADER, LINE
from beagles.backend.net.framework import Framework
from beagles.backend.darknet.dark import Darknet
from beagles.backend.io.loader import Loader
from beagles.io.logs import get_logger
from beagles.io.flags import SharedFlagIO
from beagles.backend.net.hyperparameters.cyclic_learning_rate import cyclic_learning_rate


old_graph_msg = 'Resolving old graph def {} (no guarantee)'


class TFNet:
    _TRAINER = dict({
        'rmsprop': tf.compat.v1.train.RMSPropOptimizer,
        'adadelta': tf.compat.v1.train.AdadeltaOptimizer,
        'adagrad': tf.compat.v1.train.AdagradOptimizer,
        'adagradDA': tf.compat.v1.train.AdagradDAOptimizer,
        'momentum': tf.compat.v1.train.MomentumOptimizer,
        'nesterov': tf.compat.v1.train.MomentumOptimizer,
        'adam': tf.compat.v1.train.AdamOptimizer,
        'AMSGrad': tf.compat.v1.keras.optimizers.Adam,
        'ftrl': tf.compat.v1.train.FtrlOptimizer,
        'sgd': tf.compat.v1.train.GradientDescentOptimizer
    })

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
        inp_size = [None] + self.meta['inp_size']
        self.inp = tf.compat.v1.placeholder(tf.compat.v1.float32, inp_size, 'input')
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
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
        })

        utility = min(self.flags.gpu, 1.)
        if utility > 0.0:
            self.logger.info('GPU mode with {} usage'.format(utility))
            cfg['gpu_options'] = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=utility)
            cfg['allow_soft_placement'] = True
        else:
            self.logger.info('Running entirely on CPU')
            cfg['device_count'] = {'GPU': 0}

        if self.flags.train:
            self.build_train_op()

        if self.flags.summary:
            self.summary_op = tf.compat.v1.summary.merge_all()
            self.writer = tf.compat.v1.summary.FileWriter(
                self.flags.summary + self.flags.project_name)

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(**cfg))
        # uncomment next 3 lines to enable tb debugger
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess,
        #                                                     'localhost:6064')
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if not self.ntrain:
            return
        try:
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            if self.flags.load != 0:
                self.load_from_ckpt()
        except tf.errors.NotFoundError as e:
            self.flags.error = str(e.message)
            self.io.send_flags()
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
        def _l2_norm(t):
            t = tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            return t
        self.framework.loss(self.out)
        self.logger.info('Building {} train op'.format(self.meta['model']))
        self.global_step = tf.compat.v1.Variable(0, trainable=False)
        # setup kwargs for trainer
        kwargs = dict()
        if self.flags.trainer in ['momentum', 'rmsprop', 'nesterov']:
            kwargs.update({'momentum': self.flags.momentum})
        if self.flags.trainer == 'nesterov':
            kwargs.update({'use_nesterov': True})
        if self.flags.trainer == 'AMSGrad':
            kwargs.update({'amsgrad': True})

        # setup trainer
        step_size = int(self.flags.step_size_coefficient *
                        (len(self.annotation_data) // self.flags.batch))
        self.optimizer = self._TRAINER[self.flags.trainer]\
            (
                cyclic_learning_rate(
                    self,
                    global_step=self.global_step,
                    mode=self.flags.clr_mode,
                    step_size=step_size,
                    learning_rate=self.flags.lr,
                    max_lr=self.flags.max_lr
                ), **kwargs
            )

        # setup gradients
        self.gradients, self.variables = zip(
            *self.optimizer.compute_gradients(self.framework.loss))
        if self.flags.clip:
            self.gradients, _ = tf.clip_by_global_norm(self.gradients, self.flags.clip_norm)
        # create histogram summaries
        for grad, var in zip(self.gradients, self.variables):
            name = var.name.split('/')
            with tf.compat.v1.variable_scope(name[0] + '/'):
                normed_gradients = _l2_norm(grad) if not self.flags.clip else grad
                tf.summary.histogram("gradients/" + name[1], normed_gradients)
                # tf.summary.histogram("variables/" + name[1], _l2_norm(var))

        # create train opt
        self.train_op = self.optimizer.apply_gradients(
            zip(self.gradients, self.variables),
            global_step=self.global_step)