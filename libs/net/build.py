import os
import csv
import cv2
import json
import time
import math
import pickle
from datetime import datetime
from multiprocessing.pool import ThreadPool
from threading import Thread
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from ..dark.darknet import Darknet
from ..utils.loader import create_loader
from ..utils.flags import FlagIO

train_stats = (
    'Training statistics - '
    'Learning rate: {} '
    'Batch size: {}    '
    'Epoch number: {}  '
    'Backup every: {}  '
)
pool = ThreadPool()

old_graph_msg = 'Resolving old graph def {} (no guarantee)'


class GradientNaN(Exception):
    """Raised in cases of exploding or vanishing gradient"""
    def __init__(self, flags):
        clip = "--clip argument" if flags.cli else "'Clip Gradients' checkbox"
        option = "." if flags.clip else " or turning on gradient clipping" \
                                       " using the {}.".format(clip)
        Exception.__init__(
            self, "Looks like the neural net lost the gradient try restarting"
                  " from the last checkpoint with a lower learning rate{}".format(
                   option))


class TFNet(FlagIO):
    _TRAINER = dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'nesterov': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer
    })

    def __init__(self, flags, darknet=None):
        FlagIO.__init__(self, subprogram=True)
        speak = True if darknet is None else False

        #  Setup logging verbosity
        tf_logger = tf_logging.get_logger()
        #  remove default StreamHandler and use the tf_handler from utils.flags
        tf_logger.handlers = []
        tf_logger.addHandler(self.tf_logfile)
        if os.stat(self.tf_logfile.baseFilename).st_size > 0:
            self.tf_logfile.doRollover()
        self.flags = self.read_flags()
        self.io_flags()

        self.ntrain = 0

        if self.flags.verbalise:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            tf.logging.set_verbosity(tf.logging.DEBUG)
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.logging.set_verbosity(tf.logging.FATAL)

        if self.flags.pb_load and self.flags.meta_load:
            self.logger.info('Loading from .pb and .meta')
            self.graph = tf.Graph()
            if flags.gpu > 0.0:
                device_name = flags.gpu_name
            else:
                device_name = None
            with tf.device(device_name):
                with self.graph.as_default() as g:
                    self.build_from_pb()
            return

        if darknet is None:
            darknet = Darknet(flags)
            self.ntrain = len(darknet.layers)

        self.darknet = darknet
        args = [darknet.meta, flags]
        self.num_layer = len(darknet.layers)
        self.framework = create_framework(*args)

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

    def build_from_pb(self):
        with tf.gfile.FastGFile(self.flags.pb_load, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            graph_def,
            name=""
        )
        with open(self.flags.meta_load, 'r') as fp:
            self.meta = json.load(fp)
        self.framework = create_framework(self.meta, self.flags)

        # Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.feed = dict()  # other placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

        self.setup_meta_ops()

    def build_forward(self):

        # Placeholders
        inp_size = [None] + self.meta['inp_size']
        self.inp = tf.placeholder(tf.float32, inp_size, 'input')
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
        self.out = tf.identity(state.out, name='output')

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
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(
                self.flags.summary + self.flags.project_name)

        self.sess = tf.Session(config=tf.ConfigProto(**cfg))
        # uncomment next 3 lines to enable tb debugger
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess,
        #                                                     'localhost:6064')
        self.sess.run(tf.global_variables_initializer())

        if not self.ntrain:
            return
        try:
            self.saver = tf.train.Saver(tf.global_variables(),
                                        max_to_keep=self.flags.keep)
            if self.flags.load != 0:
                self.load_from_ckpt()
        except tf.errors.NotFoundError as e:
            self.flags.error = str(e.message)
            self.send_flags()
            raise

        if self.flags.summary:
            self.writer.add_graph(self.sess.graph)

    def freeze(self):
        """
        Create a standalone const graph def that
        C++	can load and run.
        """
        darknet_ckpt = self.darknet

        with self.graph.as_default():
            for var in tf.global_variables():
                name = var.name.split(':')[0]
                var_name = name.split('-')
                l_idx = int(var_name[0])
                w_sig = var_name[1].split('/')[-1]
                l = darknet_ckpt.layers[l_idx]
                l.w[w_sig] = var.eval(self.sess)

        for layer in darknet_ckpt.layers:
            for ph in layer.h:
                layer.h[ph] = None

        flags_pb = self.flags
        flags_pb.verbalise = False
        flags_pb.train = False
        self.flags.progress = 25
        self.logger.info("Reinitializing with static TFNet...")
        tfnet_pb = TFNet(flags_pb, darknet_ckpt)
        tfnet_pb.sess = tf.Session(graph=tfnet_pb.graph)
        # tfnet_pb.predict() # uncomment for unit testing
        name = 'built_graph/{}.pb'.format(self.meta['name'])
        self.flags.progress = 50
        # Save dump of everything in meta
        with open('./data/built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
            json.dump(self.meta, fp)
        fp.close()
        self.logger.info('Saving const graph def to {}'.format(name))
        graph_def = tfnet_pb.sess.graph_def
        tf.train.write_graph(graph_def, './data/', name, False)
        self.flags.progress = 90
        self.flags.done = True

    def _save_ckpt(self, step, loss_profile):
        file = '{}-{}{}'
        model = self.meta['name']

        profile = file.format(model, step, '.profile')
        profile = os.path.join(self.flags.backup, profile)
        with open(profile, 'wb') as profile_ckpt:
            pickle.dump(loss_profile, profile_ckpt)

        ckpt = file.format(model, step, '')
        ckpt = os.path.join(self.flags.backup, ckpt)
        self.logger.info('Checkpoint at step {}'.format(step))
        self.saver.save(self.sess, ckpt)

    def train(self):
        self.io_flags()
        loss_ph = self.framework.placeholders
        loss_mva = None
        profile = list()
        goal = None
        total_steps = None
        step_pad = None
        batches = self.framework.shuffle()
        loss_op = self.framework.loss

        for i, (x_batch, datum) in enumerate(batches):
            self.flags = self.read_flags()
            feed_dict = {
                loss_ph[key]: datum[key]
                for key in loss_ph}
            feed_dict[self.inp] = x_batch
            feed_dict.update(self.feed)
            fetches = [self.train_op, loss_op]
            if self.flags.summary:
                fetches.append(self.summary_op)

            # Start the session
            try:
                fetched = self.sess.run(fetches, feed_dict)
            except tf.errors.OpError as oe:
                if oe.error_code == 3 and "nan" in oe.message.lower():
                    try:
                        raise GradientNaN(self.flags)
                    except GradientNaN as e:
                        form = "{}\nOriginal Tensorflow Error: {}"
                        self.flags.error = form.format(str(e), oe.message)
                        self.logger.error(str(e))
                        self.send_flags()
                        raise
                self.flags.error = str(oe.message)
                self.send_flags()
                raise
            loss = fetched[1]

            # single shot calculations
            if not i:
                self.logger.info(train_stats.format(
                    self.flags.lr, self.flags.batch,
                    self.flags.epoch, self.flags.save
                ))
                count = 0
            if not goal:
                goal = self.flags.size * self.flags.epoch
            if not total_steps:
                total_steps = goal // self.flags.batch
                step_pad = len(str(total_steps))
            if not loss_mva:
                loss_mva = loss

            # Check for exploding/vanishing gradient
            if math.isnan(loss) or math.isinf(loss):
                try:
                    raise GradientNaN(self.flags)
                except GradientNaN as e:
                    self.flags.error = str(e)
                    self.logger.error(str(e))
                    self.send_flags()
                    raise

            loss_mva = .9 * loss_mva + .1 * loss
            step_now = self.flags.load + i + 1

            assign_op = self.global_step.assign(step_now)
            self.sess.run(assign_op)

            # Calculate and send progress
            # noinspection PyUnboundLocalVariable
            count += self.flags.batch
            self.flags.progress = count / goal * 100
            self.io_flags()

            if self.flags.summary:
                self.writer.add_summary(fetched[2], step_now)

            form = 'step {} - loss {} - moving ave loss {} - progress {}'
            self.logger.info(
                form.format(str(step_now).zfill(step_pad),
                            format(loss, '.14f'),
                            format(loss_mva, '.14f'),
                            "{:=6.2f}%".format(self.flags.progress)))
            profile += [(loss, loss_mva)]

            ckpt = (i + 1) % (self.flags.save // self.flags.batch)
            args = [step_now, profile]

            if not ckpt:
                self._save_ckpt(*args)

        # noinspection PyUnboundLocalVariable
        if ckpt:
            # noinspection PyUnboundLocalVariable
            self._save_ckpt(*args)

    def return_predict(self, im):
        assert isinstance(im, np.ndarray), \
            'Image is not a np.ndarray'
        h, w, _ = im.shape
        im = self.framework.resize_input(im)
        this_inp = np.expand_dims(im, 0)
        feed_dict = {self.inp: this_inp}

        out = self.sess.run(self.out, feed_dict)[0]
        boxes = self.framework.findboxes(out)
        threshold = self.flags.threshold
        boxesInfo = list()
        for box in boxes:
            tmpBox = self.framework.process_box(box, h, w, threshold)
            if tmpBox is None:
                continue
            boxesInfo.append({
                "label": tmpBox[4],
                "confidence": tmpBox[6],
                "topleft": {
                    "x": tmpBox[0],
                    "y": tmpBox[2]},
                "bottomright": {
                    "x": tmpBox[1],
                    "y": tmpBox[3]}
            })
        return boxesInfo

    def predict(self):
        self.flags = self.read_flags()
        inp_path = self.flags.imgdir
        all_inps = os.listdir(inp_path)
        all_inps = [i for i in all_inps if self.framework.is_inp(i)]
        if not all_inps:
            msg = 'Failed to find any images in {} .'
            exit('Error: {}'.format(msg.format(inp_path)))

        batch = min(self.flags.batch, len(all_inps))

        # predict in batches
        n_batch = int(math.ceil(len(all_inps) / batch))
        for j in range(n_batch):
            self.logger.info(range(n_batch))
            from_idx = j * batch
            to_idx = min(from_idx + batch, len(all_inps))

            # collect images input in the batch
            this_batch = all_inps[from_idx:to_idx]
            inp_feed = pool.map(lambda inp: (
                np.expand_dims(self.framework.preprocess(
                    os.path.join(inp_path, inp)), 0)), this_batch)

            # Feed to the net
            feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
            self.logger.info('Forwarding {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            out = self.sess.run(self.out, feed_dict)
            stop = time.time()
            last = stop - start
            self.logger.info('Total time = {}s / {} inps = {} ips'.format(
                last, len(inp_feed), len(inp_feed) / last))

            # Post processing
            self.logger.info(
                'Post processing {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            pool.map(lambda p: (lambda i, prediction:
                                self.framework.postprocess(
                                    prediction,
                                    os.path.join(inp_path, this_batch[i])))(
                *p),
                     enumerate(out))
            stop = time.time()
            last = stop - start

            # Timing
            self.logger.info('Total time = {}s / {} inps = {} ips'.format(
                last, len(inp_feed), len(inp_feed) / last))

    def build_train_op(self):
        def _l2_norm(t):
            t = tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            return t
        self.framework.loss(self.out)
        self.logger.info('Building {} train op'.format(self.meta['model']))
        self.global_step = tf.Variable(0, trainable=False)

        # setup kwargs for trainer
        kwargs = dict()
        if self.flags.trainer in ['momentum', 'rmsprop', 'nesterov']:
            kwargs.update({'momentum': self.flags.momentum})
        if self.flags.trainer == 'nesterov':
            kwargs.update({'use_nesterov': True})

        # setup trainer
        step_size = int(self.flags.step_size_coefficient *
                        (len(self.framework.parse()) // self.flags.batch))
        self.optimizer = self._TRAINER[self.flags.trainer](
            self.cyclic_learning_rate(
                global_step=self.global_step,
                mode=self.flags.clr_mode,
                step_size=step_size,
                learning_rate=self.flags.lr,
                max_lr=self.flags.max_lr), **kwargs)

        # setup gradients
        self.gradients, self.variables = zip(
            *self.optimizer.compute_gradients(self.framework.loss))
        if self.flags.clip:
            self.gradients, _ = tf.clip_by_global_norm(self.gradients,
                                                    self.flags.clip_norm)
        # create histogram summaries
        for grad, var in zip(self.gradients, self.variables):
            name = var.name.split('/')
            with tf.variable_scope(name[0] + '/'):
                tf.summary.histogram("gradients/" + name[1], _l2_norm(grad))
                # tf.summary.histogram("variables/" + name[1], _l2_norm(var))

        # create train op
        self.train_op = self.optimizer.apply_gradients(
            zip(self.gradients, self.variables),
            global_step=self.global_step)

    def load_from_ckpt(self):
        if self.flags.load < 0:  # load lastest ckpt
            with open(os.path.join(self.flags.backup, 'checkpoint'), 'r') as f:
                last = f.readlines()[-1].strip()
                load_point = last.split(' ')[1]
                load_point = load_point.split('"')[1]
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

        for var in tf.global_variables():
            name = var.name.split(':')[0]
            args = [name, var.get_shape()]
            val = ckpt_loader(args)
            try:
                assert val is not None, \
                    'Cannot find and load {}'.format(var.name)
            except AssertionError as e:
                self.flags.error = str(e)
                self.logger.error(str(e))
                self.send_flags()
                raise
            shp = val.shape
            plh = tf.placeholder(tf.float32, shp)
            op = tf.assign(var, plh)
            self.sess.run(op, {plh: val})

    def camera_compile(self, cmdstring):
        cmdlist = []
        for n in self.flags.capdevs:
            cmdlist.append(compile(cmdstring.format(n), 'cmd_{}'.format(n),
                                   'exec'))
        return cmdlist

    def camera_exec(self, cmdlist):
        localdict = {'cv2': cv2, 'os': os, 'self': self, 'c': None}
        for cmd in cmdlist:
            exec(cmd, globals(), localdict)

    def camera(self):
        '''
        capture and annotate a list of devices
        number of frames displayed scales with the number of devices
        '''

        self.logger.info("Compiling capture code blocks")
        start = time.time()
        get_caps = self.camera_compile(
            "global cap{0}\n"
            "cap{0} = cv2.VideoCapture({0})\n"
            "cap{0}.set(cv2.CAP_PROP_FRAME_WIDTH, 144)\n"
            "cap{0}.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)\n"
            "cap{0}.set(cv2.CAP_PROP_BUFFERSIZE, 3)\n"
            "global annotation{0}\n"
            "annotation{0} = os.path.join("
            "self.flags.imgdir, 'video{0}_annotations.csv')")
        get_frames = self.camera_compile(
            "global ret{0}\n"
            "global frame{0}\n"
            "global stopped{0}\n"
            "ret{0}, frame{0} = cap{0}.read()\n"
            "stopped{0} = False")
        # get boxing and convert to 3-channel grayscale
        get_boxing = self.camera_compile(
            'if ret{0}:\n'
            '    global res{0}\n'
            '    global new_frame{0}\n'
            '    if self.flags.grayscale:\n'
            '        frame{0} = cv2.cvtColor(frame{0}, cv2.COLOR_BGR2GRAY)\n'
            '        frame{0} = cv2.cvtColor(frame{0}, cv2.COLOR_GRAY2BGR)\n'
            '    frame{0} = np.asarray(frame{0})\n'
            '    res{0} = self.return_predict(frame{0})\n'
            '    new_frame{0} = self.draw_box(frame{0}, res{0})\n'
            '    self.write_annotations(annotation{0}, res{0})\n')
        init_writer = self.camera_compile(
            'global out{0}\n'
            'fourcc = cv2.VideoWriter_fourcc(*"mp4v")\n'
            'max_x = cap{0}.get(cv2.CAP_PROP_FRAME_WIDTH)\n'
            'max_y = cap{0}.get(cv2.CAP_PROP_FRAME_HEIGHT)\n'
            'out{0} = cv2.VideoWriter(os.path.splitext(annotation{0})[0] + ".avi",'
            'fourcc, self.flags.fps, (int(max_x), int(max_y)))')
        write_frame = self.camera_compile('out{0}.write(new_frame{0})')
        show_frame = self.camera_compile('cv2.imshow("Cam {0}", new_frame{0})')
        end = time.time()
        self.logger.info("Finished in {}s".format(end - start))

        self.camera_exec(get_caps)
        self.logger.info("recording at {} FPS".format(self.flags.fps))
        self.camera_exec(init_writer)
        begin = time.time()
        timeout = begin + self.flags.timeout
        self.logger.info("Camera capture started on devices {}".format(self.flags.capdevs))
        while True:
            for i in [get_frames, get_boxing, write_frame, show_frame]:
                t = Thread(target=self.camera_exec(i))
                t.start()
                t.join()
            self.flags.progress = 100 * (time.time() - begin)/(timeout - begin)
            self.send_flags()
            if cv2.waitKey(1) and time.time() >= timeout:
                self.logger.info("Camera capture done on devices {}".format(
                                 self.flags.capdevs))
                break
        cv2.destroyAllWindows()

    def cyclic_learning_rate(self,
                             global_step,
                             learning_rate=0.01,
                             max_lr=0.1,
                             step_size=20.,
                             gamma=0.99994,
                             mode='triangular',
                             name=None):
        """Applies cyclic learning rate (CLR).
         From the paper:
         Smith, Leslie N. "Cyclical learning
         rates for training neural networks." 2017.
         [https://arxiv.org/pdf/1506.01186.pdf]
          This method lets the learning rate cyclically
         vary between reasonable boundary values
         achieving improved classification accuracy and
         often in fewer iterations.
          This code varies the learning rate linearly between the
         minimum (learning_rate) and the maximum (max_lr).
          It returns the cyclic learning rate. It is computed as:
           ```python
           cycle = floor( 1 + global_step /
            ( 2 * step_size ) )
          x = abs( global_step / step_size – 2 * cycle + 1 )
          clr = learning_rate +
            ( max_lr – learning_rate ) * max( 0 , 1 - x )
           ```
          Polices:
            'triangular':
              Default, linearly increasing then linearly decreasing the
              learning rate at each cycle.
             'triangular2':
              The same as the triangular policy except the learning
              rate difference is cut in half at the end of each cycle.
              This means the learning rate difference drops after each cycle.
             'exp_range':
              The learning rate varies between the minimum and maximum
              boundaries and each boundary value declines by an exponential
              factor of: gamma^global_step.
           Example: 'triangular2' mode cyclic learning rate.
            '''python
            ...
            global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=
              clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
            train_op = self.optimizer.minimize(loss_op, global_step=global_step)
            ...
             with tf.Session() as sess:
                sess.run(init)
                for step in range(1, num_steps+1):
                  assign_op = global_step.assign(step)
                  sess.run(assign_op)
            ...
             '''
           Args:
            global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
              Global step to use for the cyclic computation.  Must not be negative.
            learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate which is the lower bound
              of the cycle (default = 0.1).
            max_lr:  A scalar. The maximum learning rate boundary.
            step_size: A scalar. The number of iterations in half a cycle.
              The paper suggests step_size = 2-8 x training iterations in epoch.
            gamma: constant in 'exp_range' mode:
              gamma**(global_step)
            mode: one of {triangular, triangular2, exp_range}.
                Default 'triangular'.
                Values correspond to policies detailed above.
            name: String.  Optional name of the operation.  Defaults to
              'CyclicLearningRate'.
           Returns:
            A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
            learning rate.
          Raises:
            ValueError: if `global_step` is not supplied.
          @compatibility(eager)
          When eager execution is enabled, this function returns
          a function which in turn returns the decayed learning
          rate Tensor. This can be useful for changing the learning
          rate value across different invocations of self.optimizer functions.
          @end_compatibility
      """
        if global_step is None:
            raise ValueError(
                "global_step is required for cyclic_learning_rate.")
        with ops.name_scope(name, os.path.basename(self.flags.model),
                            [learning_rate, global_step]) as name:
            learning_rate = ops.convert_to_tensor(learning_rate,
                                                  name="learning_rate")
            dtype = learning_rate.dtype
            global_step = math_ops.cast(global_step, dtype)
            step_size = math_ops.cast(step_size, dtype)

            def cyclic_lr():
                """Helper to recompute learning rate; most helpful in eager-mode."""
                # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
                double_step = math_ops.multiply(2., step_size)
                global_div_double_step = math_ops.divide(global_step,
                                                         double_step)
                cycle = math_ops.floor(
                    math_ops.add(1., global_div_double_step))
                # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
                double_cycle = math_ops.multiply(2., cycle)
                global_div_step = math_ops.divide(global_step, step_size)
                tmp = math_ops.subtract(global_div_step, double_cycle)
                x = math_ops.abs(math_ops.add(1., tmp))
                # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
                a1 = math_ops.maximum(0., math_ops.subtract(1., x))
                a2 = math_ops.subtract(max_lr, learning_rate)
                clr = math_ops.multiply(a1, a2)
                if mode == 'triangular2':
                    clr = math_ops.divide(clr, math_ops.cast(
                        math_ops.pow(2, math_ops.cast(
                            cycle - 1, tf.int32)), tf.float32))
                if mode == 'exp_range':
                    clr = math_ops.multiply(
                        math_ops.pow(gamma, global_step), clr)
                return math_ops.add(clr, learning_rate, name=name)

            if not context.executing_eagerly():
                cyclic_lr = cyclic_lr()
            tf.summary.scalar("/".join([self.flags.trainer,
                                        'cyclic_learning_rate']), cyclic_lr)
            return cyclic_lr

    def draw_box(self, original_img, predictions):
        """
        Args:
            original_img: A numpy ndarray
            predictions: A nested dictionary object of the form
                        {"label": str, "confidence": float,
                        "topleft": {"x": int, "y": int},
                        "bottomright": {"x": int, "y": int}}
        Returns:
            A numpy ndarray with boxed detections
        """
        new_image = np.copy(original_img)

        for result in predictions:

            confidence = result['confidence']

            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            header = " ".join([result['label'], str(round(confidence, 3))])

            if confidence > self.flags.threshold:
                new_image = cv2.rectangle(new_image, (top_x, top_y),
                                          (btm_x, btm_y), (255, 0, 0), 3)
                new_image = cv2.putText(new_image, header, (top_x, top_y - 5),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                        (0, 230, 0), 1, cv2.LINE_AA)
        return new_image

    def write_annotations(self, annotation_file, prediction):

        def _center(x1, y1, x2, y2):
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            return x, y

        with open(annotation_file, mode='a') as file:
            file_writer = csv.writer(file, delimiter=',',
                                     quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            for result in prediction:
                if result['confidence'] > self.flags.threshold:

                    center_x, center_y = _center(result['topleft']['x'],
                                                 result['topleft']['y'],
                                                 result['bottomright']['x'],
                                                 result['bottomright']['y'])

                    file_writer.writerow([datetime.now(),
                                         result['label'],
                                         result['confidence'],
                                         center_x,
                                         center_y,
                                         result['topleft']['x'],
                                         result['topleft']['y'],
                                         result['bottomright']['x'],
                                         result['bottomright']['y']])

    def annotate(self):
        INPUT_VIDEO = self.flags.fbf
        FRAME_NUMBER = 0
        cap = cv2.VideoCapture(INPUT_VIDEO)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        annotation_file = os.path.splitext(INPUT_VIDEO)[0] + '_annotations.csv'

        if os.path.exists(annotation_file):
            self.logger.info("Overwriting existing annotations")
            os.remove(annotation_file)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        max_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        max_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        out = cv2.VideoWriter(os.path.splitext(INPUT_VIDEO)[0] + '_annotated.avi',
                              fourcc, 20.0, (int(max_x), int(max_y)))
        self.logger.info('Annotating ' + INPUT_VIDEO)

        while True:  # Capture frame-by-frame
            FRAME_NUMBER += 1
            ret, frame = cap.read()
            if ret:
                self.flags.progress = round((100 * FRAME_NUMBER / total_frames), 0)
                if FRAME_NUMBER % 10 == 0:
                    self.io_flags()
                frame = np.asarray(frame)
                result = self.return_predict(frame)
                new_frame = self.draw_box(frame, result)
                self.write_annotations(annotation_file, result)
                out.write(new_frame)
                if self.flags.kill:
                    break
            else:
                break
        # When everything done, release the capture
        out.release()

        # file = self.flags.demo  # TODO add asynchronous capture
        # save_video = self.flags.save_video
        #
        # if file == 'camera':
        #     file = 0
        # else:
        #     assert os.path.isfile(file), \
        #         'file {} does not exist'.format(file)
        #
        # camera = cv2.VideoCapture(file)
        #
        # if file == 0:
        #     self.logger.info('Press [ESC] to quit demo')
        #
        # assert camera.isOpened(), \
        #     'Cannot capture source'
        #
        # if file == 0:  # camera window
        #     cv2.namedWindow('', 0)
        #     _, frame = camera.read()
        #     max_y, max_x, _ = frame.shape
        #     cv2.resizeWindow('', max_x, max_y)
        # else:
        #     _, frame = camera.read()
        #     max_y, max_x, _ = frame.shape
        #
        # if save_video:
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     if file == 0:  # camera window
        #         fps = 1 / self._get_fps(frame)
        #         if fps < 1:
        #             fps = 1
        #     else:
        #         fps = round(camera.get(cv2.CAP_PROP_FPS))
        #     videoWriter = cv2.VideoWriter(
        #         self.flags.save_video, fourcc, fps, (max_x, max_y))
        #
        # # buffers for demo in batch
        # buffer_inp = list()
        # buffer_pre = list()
        #
        # elapsed = int()
        # start = timer()
        # # Loop through frames
        # while camera.isOpened():
        #     elapsed += 1
        #     _, frame = camera.read()
        #     if frame is None:
        #         print('\nEnd of Video')
        #         break
        #     preprocessed = self.framework.preprocess(frame)
        #     buffer_inp.append(frame)
        #     buffer_pre.append(preprocessed)
        #
        #     # Only process and imshow when queue is full
        #     if elapsed % self.flags.queue == 0:
        #         feed_dict = {self.inp: buffer_pre}
        #         net_out = self.sess.run(self.out, feed_dict)
        #         for img, single_out in zip(buffer_inp, net_out):
        #             postprocessed = self.framework.postprocess(
        #                 single_out, img, False)
        #             if save_video:
        #                 videoWriter.write(postprocessed)
        #             if file == 0:  # camera window
        #                 cv2.imshow('', postprocessed)
        #         # Clear Buffers
        #         buffer_inp = list()
        #         buffer_pre = list()
        #
        #     if elapsed % 5 == 0:
        #         sys.stdout.write('\r')
        #         sys.stdout.write('{0:3.3f} FPS'.format(
        #             elapsed / (timer() - start)))
        #         sys.stdout.flush()
        #     if file == 0:  # camera window
        #         choice = cv2.waitKey(1)
        #         if choice == 27: break
        #
        # sys.stdout.write('\n')
        # if save_video:
        #     videoWriter.release()
        # camera.release()
        # if file == 0:  # camera window
        #     cv2.destroyAllWindows()

        # def _get_fps(self, frame):
        #     elapsed = int()
        #     start = time.time()
        #     preprocessed = self.framework.preprocess(frame)
        #     feed_dict = {self.inp: [preprocessed]}
        #     net_out = self.sess.run(self.out, feed_dict)[0]
        #     processed = self.framework.postprocess(net_out, frame, False)
        #     return time.time() - start
