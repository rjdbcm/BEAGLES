import os
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from libs.backend.net.ops import op_create, identity
from libs.backend.net.ops.baseop import HEADER, LINE
from libs.backend.net.framework import Framework
from libs.backend.dark.darknet import Darknet
from libs.utils.loader import create_loader
from libs.io.flags import FlagIO
from libs.utils.errors import *
from libs.backend.net.hyperparameters.cyclic_learning_rate import cyclic_learning_rate


old_graph_msg = 'Resolving old graph def {} (no guarantee)'


class TFNet(FlagIO):
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
        FlagIO.__init__(self, subprogram=True)
        speak = True if darknet is None else False
        # disable eager mode for TF1-dependent code
        tf.compat.v1.disable_eager_execution()

        #  Setup logging verbosity
        tf_logger = tf_logging.get_logger()
        #  remove default StreamHandler and use the tf_handler from utils.flags
        tf_logger.handlers = []
        tf_logger.addHandler(self.tf_logfile)
        if os.stat(self.tf_logfile.baseFilename).st_size > 0:
            self.tf_logfile.doRollover()
        self.flags = self.read_flags() if self.read_flags() is not None else flags
        self.io_flags()

        self.ntrain = 0

        if self.flags.verbalise:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

        if darknet is None:
            darknet = Darknet(flags)
            self.ntrain = len(darknet.layers)

        self.darknet = darknet
        args = [darknet.meta, flags]
        self.num_layer = len(darknet.layers)
        self.framework = Framework.create(*args)
        self.data = self.framework.parse()


        self.meta = darknet.meta
        if speak:
            self.logger.info('Building net ...')
        start = time.time()
        self.graph = tf.Graph()
        if flags.gpu > 0.0:
            device_name = flags.gpu_name
        else:
            device_name = None
        with tf.device(device_name):
            with self.graph.as_default():
                self.build_forward()
                self.setup_meta_ops()
        self.logger.info('Finished in {}s'.format(
            time.time() - start))

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
            self.send_flags()
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
            if mess:
                self.logger.info(mess)
            else:
                self.logger.info(LINE)
        self.logger.info(LINE)

        self.top = state
        self.out = tf.compat.v1.identity(state.out, name='output')

    def setup_meta_ops(self):
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
        })

        utility = min(self.flags.gpu, 1.)
        if utility > 0.0:
            self.logger.info('GPU mode with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(
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
        ckpt_loader = create_loader(ckpt)
        self.logger.info(old_graph_msg.format(ckpt))

        for var in tf.compat.v1.global_variables():
            name = var.name.split(':')[0]
            args = [name, var.get_shape()]
            val = ckpt_loader(args)
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
                        (len(self.data) // self.flags.batch))
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
            self.gradients, _ = tf.clip_by_global_norm(self.gradients,
                                                       self.flags.clip_norm)
        # create histogram summaries
        for grad, var in zip(self.gradients, self.variables):
            name = var.name.split('/')
            with tf.compat.v1.variable_scope(name[0] + '/'):
                tf.summary.histogram("gradients/" + name[1], _l2_norm(grad))
                # tf.summary.histogram("variables/" + name[1], _l2_norm(var))

        # create train op
        self.train_op = self.optimizer.apply_gradients(
            zip(self.gradients, self.variables),
            global_step=self.global_step)