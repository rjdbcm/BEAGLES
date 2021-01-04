import os
import cv2
from random import random
from typing import SupportsInt, List, SupportsFloat
from collections import Iterator
from beagles.backend.io.pascal_voc_clean_xml import pascal_voc_clean_xml
from copy import deepcopy
import tensorflow as tf
import numpy as np
_np = np # always the original numpy API
try: # falls back to numpy if tensorflow version is less than 2.4.0
    import tensorflow.experimental.numpy as tnp
    TF_NUMPY = True
except AttributeError:
    TF_NUMPY = False
np = tnp if TF_NUMPY else np

def is_input(self, name):
    """Checks if input is a valid image file"""
    return cv2.haveImageReader(name)

def parse(self, exclusive=False):
    meta = self.meta
    ann = self.flags.annotation
    if not os.path.isdir(ann):
        raise NotADirectoryError(f'Annotation directory not found {ann}')
    dumps, weights = pascal_voc_clean_xml(self, ann, meta['labels'], exclusive)
    return dumps, weights


def batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    # sizes are passed to avoid having separate get_feed_values methods for
    # YOLO and YOLOv2
    S = self.meta['side']
    return self.get_feed_values(chunk, S, S)


def get_preprocessed(self, chunk):
    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.flags.dataset, jpg)
    img, allobj = self.preprocess(path, allobj)
    return img, w, h, allobj


def get_feed_values(self, chunk, dim1, dim2):

    H = dim1
    W = dim2
    C = self.meta['classes']
    B = self.meta['num']
    labels = self.meta['labels']

    # preprocess
    img, w, h, allobj = self.get_preprocessed(chunk)

    if not allobj:
        allobj = []

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
        centery = .5 * (obj[2] + obj[4])  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H:
            return None, None
        obj[3] = float(obj[3] - obj[1]) / w
        obj[4] = float(obj[4] - obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx)  # centerx
        obj[2] = cy - np.floor(cy)  # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = _np.zeros([H * W, B, C])
    confs = _np.zeros([H * W, B])
    coord = _np.zeros([H * W, B, 4])
    proid = _np.zeros([H * W, B, C])
    prear = _np.zeros([H * W, 4])

    for obj in allobj:
        probs[obj[5], :, :] = [[0.] * C] * B
        probs[obj[5], :, labels.index(obj[0])] = 1.
        proid[obj[5], :, :] = [[1.] * C] * B
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * W  # xleft
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * H  # yup
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * W  # xright
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * H  # ybot
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        '_probs': tf.cast(probs, tf.float32), '_confs': tf.cast(confs, tf.float32),
        '_coord': tf.cast(coord, tf.float32), '_proid': tf.cast(proid, tf.float32),
        '_areas': tf.cast(areas, tf.float32), '_upleft': tf.cast(upleft, tf.float32),
        '_botright': tf.cast(botright, tf.float32)
    }

    return inp_feed_val, loss_feed_val

class Sample(Iterator):
    """
    an iterator class that returns k random samples from another iterator
    that has no random-access. Requires: the iterator length as an input

    similar to random.sample and can be overloaded to work like numpy.random.choice
    Args:

        seq: iterator object
        n: size of the sequence object
        k: output sequence size (number of samples in the subset)
        p: probabilities associated with seq

    """

    def __init__(self, seq: Iterator, n: int, k: int, p: List[float]=None):
        self.seq = seq
        self.N = n
        self._N = n
        self.K = k
        if p is not None:
            if len(p) != n:
                raise IndexError("The len(p) must be equal to n")
            if np.sum(p) != 1.0:
                raise IndexError("The sum(p) must be equal to 1.0")
            self.p = p
        else:
            self.p = [1.0] * self._N

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            r = random()
            item = next(self.seq)
            idx = self._N - self.N if self.N > 0 else self._N - 1
            if r**self.p[idx] <= float(self.K) / self.N:
                self.K -= 1
                self.N -= 1
                return item
            else:
                self.N -= 1

def shuffle(self, data, class_weights=None):
    self.flags.size = size = len(data)
    _batch = self.flags.batch
    self.logger.info(f'Dataset of {size} instances')
    if _batch > size:
        _batch = size
    try:
        batch_per_epoch = int(size / _batch)
    except ZeroDivisionError:
        self.flags.error = "No image data to shuffle."
        self.logger.error("No image data to shuffle.")
        self.flags.kill = True
        self.io.send_flags()
        raise
    probs = None
    if class_weights is not None:
        probs = list()
        score = list()
        for i in data:
            for box in i[1][2]:
                score.append(class_weights.get(str(box[0])))
            probs.append(np.subtract(1, np.mean(score)))
        probs = np.divide(probs, np.sum(probs))

    for e in range(self.flags.epoch):
        for b in range(batch_per_epoch):
            feed = dict()
            x_batch = list()
            sample = Sample(iter(data), size, _batch, probs)
            for j in sample:
                try:
                    inp, new_feed = self.batch(j)
                except ZeroDivisionError:
                    self.logger.error("This image's width or height are zeros: ", j[0])
                    self.logger.error('train_instance:', j)
                    self.logger.error('Please remove or fix it then try again.')
                    raise
                if inp is None:
                    continue
                x_batch += [np.expand_dims(inp, 0)]
                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed.get(key, np.zeros((0,) + new.shape))
                    feed[key] = np.concatenate([old_feed, [new]])
            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed.values()
        self.logger.info(f'Finish epoch {e + 1} of {self.flags.epoch}')
