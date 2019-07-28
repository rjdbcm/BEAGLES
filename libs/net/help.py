"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import sys
import csv
import cv2
import os

old_graph_msg = 'Resolving old graph def {} (no guarantee)'


def build_train_op(self):
    self.framework.loss(self.out)
    self.logger.info('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.flags.trainer](self.flags.lr)
    if self.flags.clip == False:
        self.gradients = optimizer.compute_gradients(self.framework.loss)
    if self.flags.clip == True:
        # From github.com/thtrieu/darkflow/issues/557#issuecomment-377378352
        # avoid gradient explosions late in training
        self.gradients = [(tf.clip_by_value(grad, -1., 1.), var) for
                     grad, var in optimizer.compute_gradients(
                self.framework.loss)]
    self.train_op = optimizer.apply_gradients(self.gradients)


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
        load_old_graph(self, load_point)


def load_old_graph(self, ckpt):
    ckpt_loader = create_loader(ckpt)
    self.logger.info(old_graph_msg.format(ckpt))

    for var in tf.global_variables():
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        assert val is not None, \
            'Cannot find and load {}'.format(var.name)
        shp = val.shape
        plh = tf.placeholder(tf.float32, shp)
        op = tf.assign(var, plh)
        self.sess.run(op, {plh: val})


# def _get_fps(self, frame):
#     elapsed = int()
#     start = timer()
#     preprocessed = self.framework.preprocess(frame)
#     feed_dict = {self.inp: [preprocessed]}
#     net_out = self.sess.run(self.out, feed_dict)[0]
#     processed = self.framework.postprocess(net_out, frame, False)
#     return timer() - start


def camera_compile(self, cmdstring):
    cmdlist = []
    for n in self.cams:
        cmdlist.append(compile(cmdstring.format(n), 'cmd_{}'.format(n),
                               'exec'))
    return cmdlist


def camera_exec(self, cmdlist):
    localdict = {'cv2': cv2, 'os': os, 'self': self, 'c': None}
    for cmd in cmdlist:
        exec(cmd, globals(), localdict)


def camera(self):
    self.cams = self.flags.capdevs
    self.logger.info("Camera capture started on devices {}".format(self.cams))
    get_caps = self.camera_compile(
        "global cap{0}\n"
        "cap{0} = cv2.VideoCapture({0})\n"
        "cap{0}.set(cv2.CAP_PROP_FRAME_WIDTH, 144)\n"
        "cap{0}.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)\n"
        "global annotation{0}\n"
        "annotation{0} = os.path.join("
        "self.flags.imgdir, 'video{0}_annotations.csv')")
    get_frames = self.camera_compile(
        "global ret{0}\n"
        "global frame{0}\n"
        "ret{0}, frame{0} = cap{0}.read()")
    get_boxing = self.camera_compile(
        'if ret{0}:\n'
        '    global res{0}\n'
        '    global new_frame{0}\n'
        '    frame{0} = np.asarray(frame{0})\n'
        '    res{0} = self.return_predict(frame{0})\n'
        '    new_frame{0} = self.draw_box(frame{0}, res{0})\n'
        '    self.write_annotations(annotation{0}, res{0})\n'
        '    cv2.imshow("Cam {0}", new_frame{0})')
    self.camera_exec(get_caps)
    timeout = time.time() + self.flags.timeout
    while True:
        self.camera_exec(get_frames)
        self.camera_exec(get_boxing)

        if cv2.waitKey(1) and time.time() > timeout:
            self.logger.info("Camera capture done on devices {}".format(
                             self.flags.capdevs))
            break
    cv2.destroyAllWindows()

    # file = self.flags.demo  # TODO add asynchronous capture
    # SaveVideo = self.flags.saveVideo
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
    # if SaveVideo:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     if file == 0:  # camera window
    #         fps = 1 / self._get_fps(frame)
    #         if fps < 1:
    #             fps = 1
    #     else:
    #         fps = round(camera.get(cv2.CAP_PROP_FPS))
    #     videoWriter = cv2.VideoWriter(
    #         self.flags.saveVideo, fourcc, fps, (max_x, max_y))
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
    #             if SaveVideo:
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
    # if SaveVideo:
    #     videoWriter.release()
    # camera.release()
    # if file == 0:  # camera window
    #     cv2.destroyAllWindows()


def draw_box(self, original_img, predictions):
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

    with open(annotation_file, mode='a') as file:
        file_writer = csv.writer(file, delimiter=',',
                                 quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)
        for result in prediction:
            if result['confidence'] > self.flags.threshold:
                file_writer.writerow([datetime.now(),
                                     result['label'],
                                     result['confidence'],
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

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    max_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    max_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(os.path.splitext(INPUT_VIDEO)[0] + '_annotated.avi',
                          fourcc, 20.0, (int(max_x), int(max_y)))
    self.logger.info('Annotating ' + INPUT_VIDEO)

    while True:  # Capture frame-by-frame
        FRAME_NUMBER += 1
        ret, frame = cap.read()
        if ret == True:
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


def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
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

    return darknet_ckpt
