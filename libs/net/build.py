import tensorflow as tf
from tensorflow.python.platform import tf_logging
import time
from . import help
from . import flow
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from ..dark.darknet import Darknet
import json
from ..utils.flags import FlagIO
import logging
import os


class TFNet(FlagIO):

    _TRAINER = dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer
    })

    # imported methods
    _get_fps = help._get_fps
    _exec = help._exec
    _compile = help._compile
    boxing = help.boxing
    train = flow.train
    camera = help.camera
    annotate = help.annotate
    predict = flow.predict
    return_predict = flow.return_predict
    to_darknet = help.to_darknet
    build_train_op = help.build_train_op
    load_from_ckpt = help.load_from_ckpt

    def __init__(self, FLAGS, darknet=None):
        FlagIO.__init__(self, subprogram=True)
        self.ntrain = 0

        #  Setup logging verbosity
        tf_logger = tf_logging.get_logger()
        #  remove default StreamHandler and use the tf_handler from utils.flags
        tf_logger.handlers = []
        tf_logger.addHandler(self.tf_logfile)
        self.FLAGS = self.read_flags()
        self.io_flags()
        if self.FLAGS.verbalise:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            tf.logging.set_verbosity(tf.logging.DEBUG)
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.logging.set_verbosity(tf.logging.FATAL)

        if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
            self.logger.info('Loading from .pb and .meta')
            self.graph = tf.Graph()
            if FLAGS.gpu > 0.0:
                device_name = FLAGS.gpuName
            else:
                device_name = None
            with tf.device(device_name):
                with self.graph.as_default() as g:
                    self.build_from_pb()
            return

        if darknet is None:
            darknet = Darknet(FLAGS)
            self.ntrain = len(darknet.layers)

        self.darknet = darknet
        args = [darknet.meta, FLAGS]
        self.num_layer = len(darknet.layers)
        self.framework = create_framework(*args)

        self.meta = darknet.meta

        self.logger.info('Building net ...')
        start = time.time()
        self.graph = tf.Graph()
        if FLAGS.gpu > 0.0:
            device_name = FLAGS.gpuName
        else:
            device_name = None
        with tf.device(device_name):
            with self.graph.as_default() as g:
                self.build_forward()
                self.setup_meta_ops()
        self.logger.info('Finished in {}s'.format(
            time.time() - start))

    def build_from_pb(self):
        with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            graph_def,
            name=""
        )
        with open(self.FLAGS.metaLoad, 'r') as fp:
            self.meta = json.load(fp)
        self.framework = create_framework(self.meta, self.FLAGS)

        # Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.feed = dict()  # other placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

        self.setup_meta_ops()

    def build_forward(self):
        verbalise = self.FLAGS.verbalise

        # Placeholders
        inp_size = [None] + self.meta['inp_size']
        self.inp = tf.placeholder(tf.float32, inp_size, 'input')
        self.feed = dict()  # other placeholders

        # Build the forward pass
        state = identity(self.inp)
        roof = self.num_layer - self.ntrain
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
        self.out = tf.identity(state.out, name='output')

    def setup_meta_ops(self):
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
        })

        utility = min(self.FLAGS.gpu, 1.)
        if utility > 0.0:
            self.logger.info('GPU mode with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction=utility)
            cfg['allow_soft_placement'] = True
        else:
            self.logger.info('Running entirely on CPU')
            cfg['device_count'] = {'GPU': 0}

        if self.FLAGS.train:
            self.build_train_op()

        if self.FLAGS.summary:
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')

        self.sess = tf.Session(config=tf.ConfigProto(**cfg))
        self.sess.run(tf.global_variables_initializer())

        if not self.ntrain: return
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.keep)
        if self.FLAGS.load != 0:
            self.load_from_ckpt()

        if self.FLAGS.summary:
            self.writer.add_graph(self.sess.graph)

    def savepb(self):
        """
        Create a standalone const graph def that
        C++	can load and run.
        """
        
        darknet_pb = self.to_darknet()
        flags_pb = self.FLAGS
        flags_pb.verbalise = False
        flags_pb.train = False
        self.FLAGS.progress = 25
        # rebuild another tfnet. all const.
        tfnet_pb = TFNet(flags_pb, darknet_pb)
        tfnet_pb.sess = tf.Session(graph=tfnet_pb.graph)
        # tfnet_pb.predict() # uncomment for unit testing
        name = 'built_graph/{}.pb'.format(self.meta['name'])
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.FLAGS.progress = 50
        # Save dump of everything in meta
        with open('./data/built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
            json.dump(self.meta, fp)
        fp.close()
        self.logger.info('Saving const graph def to {}'.format(name))
        graph_def = tfnet_pb.sess.graph_def
        tf.train.write_graph(graph_def, './data/', name, False)
        self.FLAGS.progress = 90
        self.FLAGS.done = True
