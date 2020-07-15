from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from .predict import preprocess
# from .misc import show
from copy import deepcopy
import pickle
import numpy as np
import os 


def parse(self, exclusive=False):
    meta = self.meta
    ext = '.parsed'
    ann = self.flags.annotation
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    self.logger.info('{} parsing {}'.format(meta['model'], ann))
    dumps = pascal_voc_clean_xml(self, ann, meta['labels'], exclusive)
    return dumps


def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    # sizes are passed to avoid having separate get_feed_values methods for
    # YOLO and YOLOv2
    S = self.meta['side']
    return self.get_feed_values(chunk, S, S)


def get_preprocessed_img(self, chunk):
    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.flags.dataset, jpg)
    img = self.preprocess(path, allobj)
    return img, w, h, allobj


def get_feed_values(self, chunk, dim1, dim2):

    H = dim1
    W = dim2
    C = self.meta['classes']
    B = self.meta['num']
    labels = self.meta['labels']

    # preprocess
    img, w, h, allobj = self.get_preprocessed_img(chunk)

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
    probs = np.zeros([H * W, B, C])
    confs = np.zeros([H * W, B])
    coord = np.zeros([H * W, B, 4])
    proid = np.zeros([H * W, B, C])
    prear = np.zeros([H * W, 4])
    # TODO: use np.put instead
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
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }

    return inp_feed_val, loss_feed_val


def shuffle(self):
    batch = self.flags.batch
    data = self.parse()
    self.flags.size = len(data)
    self.io_flags()
    self.logger.info('Dataset of {} instance(s)'.format(self.flags.size))
    if batch > self.flags.size:
        self.flags.batch = batch = self.flags.size
    batch_per_epoch = int(self.flags.size / batch)

    for i in range(self.flags.epoch):
        shuffle_idx = perm(np.arange(self.flags.size))
        for b in range(batch_per_epoch):
            # yield these
            x_batch = list()
            feed_batch = dict()

            for j in range(b*batch, b*batch+batch):
                train_instance = data[shuffle_idx[j]]
                self.logger.debug(train_instance[0])
                try:
                    inp, new_feed = self._batch(train_instance)
                except ZeroDivisionError:
                    self.logger.error("This image's width or height are zeros: ", train_instance[0])
                    self.logger.error('train_instance:', train_instance)
                    self.logger.error('Please remove or fix it then try again.')
                    raise

                if inp is None:
                    continue
                x_batch += [np.expand_dims(inp, 0)]

                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key, 
                        np.zeros((0,) + new.shape))
                    feed_batch[key] = np.concatenate([ 
                        old_feed, [new] 
                    ])      
            
            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch
        
        self.logger.info('Finish {} epoch(es)'.format(i + 1))

