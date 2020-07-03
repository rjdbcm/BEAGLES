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
from ..utils.postprocess import BehaviorIndex


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
        self.flags = self.read_flags()
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

            # assign_op = global_step.assign(step_now)
            # self.sess.run(assign_op)

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
                                    os.path.join(inp_path, this_batch[i])))(*p),
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
            with tf.compat.v1.variable_scope(name[0] + '/'):
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
            try:
                assert val is not None, \
                    'Cannot find and load {}'.format(var.name)
            except AssertionError as e:
                self.flags.error = str(e)
                self.logger.error(str(e))
                self.send_flags()
                raise
            shp = val.shape
            plh = tf.compat.v1.placeholder(tf.float32, shp)
            op = tf.compat.v1.assign(var, plh)
            self.sess.run(op, {plh: val})

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

    def write_annotations(self, annotation_file, prediction, time_elapsed, epoch):

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

                    file_writer.writerow([datetime.fromtimestamp(epoch + time_elapsed),
                                         result['label'],
                                         result['confidence'],
                                         center_x,
                                         center_y,
                                         result['topleft']['x'],
                                         result['topleft']['y'],
                                         result['bottomright']['x'],
                                         result['bottomright']['y']])

    def analyze(self, file_list):

        bi = BehaviorIndex(file_list)
        if len(file_list) > 1:
            for i in file_list:
                analysis_file = os.path.splitext(i)[0] + '_analysis.json'
                with open(analysis_file, mode='a') as file:
                    file.write(bi.individual_total_beh())
                    file.write(bi.individual_single_beh())
            analysis_file = 'group_analysis.json'
            with open(analysis_file, mode='a') as file:
                file.write(bi.group_total_beh())
                file.write(bi.group_single_beh())
        else:
            analysis_file = os.path.splitext(file_list[0])[0] + '_analysis.json'
            with open(analysis_file, mode='a') as file:
                file.write(bi.individual_total_beh())
                file.write(bi.individual_single_beh())

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

        start_time = time.time()

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

                # This is a hackish way of making sure we can quantify videos
                # taken at different times
                epoch = datetime(1970, 1, 1, 0, 0).timestamp()
                time_elapsed = time.time() - start_time
                self.write_annotations(annotation_file,
                                       result,
                                       time_elapsed,
                                       epoch)
                out.write(new_frame)
                if self.flags.kill:
                    break
            else:
                break
        # When everything done, release the capture
        out.release()
