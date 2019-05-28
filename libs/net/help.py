"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
from time import time as timer
import tensorflow as tf
import numpy as np
from datetime import datetime
import sys
import csv
import cv2
import os

old_graph_msg = 'Resolving old graph def {} (no guarantee)'


def build_train_op(self):
    self.framework.loss(self.out)
    self.say('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    if self.FLAGS.clip == False:
        gradients = optimizer.compute_gradients(self.framework.loss)
    if self.FLAGS.clip == True:
        # From github.com/thtrieu/darkflow/issues/557#issuecomment-377378352 avoid gradient explosions late in training
        gradients = [(tf.clip_by_value(grad, -1., 1.), var) for
                     grad, var in optimizer.compute_gradients(self.framework.loss)]
    self.train_op = optimizer.apply_gradients(gradients)


def load_from_ckpt(self):
    if self.FLAGS.load < 0:  # load lastest ckpt
        with open(os.path.join(self.FLAGS.backup, 'checkpoint'), 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.FLAGS.load = int(load_point)

    load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
    load_point = '{}-{}'.format(load_point, self.FLAGS.load)
    self.say('Loading from {}'.format(load_point))
    try:
        self.saver.restore(self.sess, load_point)
    except:
        load_old_graph(self, load_point)


def say(self, *msgs):
    if self.FLAGS.verbalise:
        with open(self.FLAGS.log, 'a') as logfile:
                msgs = list(msgs)
                form = "[{}] {}\n"
                for msg in msgs:
                    if msg is None:
                        continue
                    else:
                        logfile.write(form.format(datetime.now(), msg))
        if self.FLAGS.killed or self.FLAGS.done:
            logfile.close()

def load_old_graph(self, ckpt):
    ckpt_loader = create_loader(ckpt)
    self.say(old_graph_msg.format(ckpt))

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


def _get_fps(self, frame):
    elapsed = int()
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame, False)
    return timer() - start


def camera(self):
    file = self.FLAGS.demo  # TODO add asynchronous capture
    SaveVideo = self.FLAGS.saveVideo

    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), \
            'file {} does not exist'.format(file)

    camera = cv2.VideoCapture(file)

    if file == 0:
        self.say('Press [ESC] to quit demo')

    assert camera.isOpened(), \
        'Cannot capture source'

    if file == 0:  # camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read()
        max_y, max_x, _ = frame.shape
        cv2.resizeWindow('', max_x, max_y)
    else:
        _, frame = camera.read()
        max_y, max_x, _ = frame.shape

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if file == 0:  # camera window
            fps = 1 / self._get_fps(frame)
            if fps < 1:
                fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            self.FLAGS.saveVideo, fourcc, fps, (max_x, max_y))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()

    elapsed = int()
    start = timer()
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print('\nEnd of Video')
            break
        preprocessed = self.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)

        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                postprocessed = self.framework.postprocess(
                    single_out, img, False)
                if SaveVideo:
                    videoWriter.write(postprocessed)
                if file == 0:  # camera window
                    cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if file == 0:  # camera window
            choice = cv2.waitKey(1)
            if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    if file == 0:  # camera window
        cv2.destroyAllWindows()


def annotate(self):
    # TODO: ignore small bounding boxes
    lb = self.FLAGS.lb
    INPUT_VIDEO = self.FLAGS.fbf
    FRAME_NUMBER = 0
    cap = cv2.VideoCapture(INPUT_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annotation_file = os.path.splitext(INPUT_VIDEO)[0] + '_annotations.csv'
    if os.path.exists(annotation_file):
        self.say("Overwriting existing annotations")
        os.remove(annotation_file)
    max_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    max_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    max_per = (2 * max_x) + (2 * max_y)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(os.path.splitext(INPUT_VIDEO)[0] + '_annotated.avi', fourcc, 20.0, (int(max_x), int(max_y)))
    self.say('Annotating ' + INPUT_VIDEO + ' press [ESC] to quit')

    def boxing(original_img, predictions):
        newImage = np.copy(original_img)

        for result in predictions:

            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            bb_width = abs(btm_x - top_x)
            bb_height = abs(top_y - btm_y)

            bb_per = (2 * bb_width) + (2 * bb_height)
            bb_pct = round((bb_per / max_per) * 100, 0)

            lb_per = lb * max_per

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3)) + " Per: " + str(bb_pct) + "%"

            if bb_per < lb_per:
                print('Ignoring\n--------\nDetection: {}\nBbox Perimeter: {}\nLower Bound: {}\n'.format(label, bb_per,
                                                                                                        lb_per))

            if confidence > 0.3 and not bb_per < lb_per:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
                newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                       (0, 230, 0), 1, cv2.LINE_AA)
                gen_annotations(predictions)

        return newImage

    def gen_annotations(predictions):
        with open(annotation_file, mode='a') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in predictions:
                time_elapsed = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                labels = item['label']
                conf = item['confidence']
                top_x = item['topleft']['x']
                top_y = item['topleft']['y']
                btm_x = item['bottomright']['x']
                btm_y = item['bottomright']['y']
                file_writer.writerow([time_elapsed, labels, conf, top_x, top_y, btm_x, btm_y])

    while True:
        # Capture frame-by-frame
        FRAME_NUMBER += 1
        ret, frame = cap.read()
        if ret == True:
            self.say(
                "Frame {}/{} [{}%]".format(FRAME_NUMBER, total_frames, round(100 * FRAME_NUMBER / total_frames),
                                           1))
            frame = np.asarray(frame)
            result = self.return_predict(frame)
            new_frame = boxing(frame, result)  # Display the resulting frame
            out.write(new_frame)
            cv2.imshow(INPUT_VIDEO, new_frame)
            choice = cv2.waitKey(1)
            if choice == 27:
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
