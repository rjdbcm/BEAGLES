import tensorflow.compat.v1.layers as slim
import pickle
import tensorflow as tf
from libs.net.yolo.misc import show
import numpy as np
import os
import math


def expit_tensor(x):
    return 1. / (1. + tf.compat.v1.exp(-x))


def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W  # number of grid cells
    anchors = m['anchors']

    self.logger.info('{} loss hyper-parameters:'.format(m['model']))
    self.logger.info('H       = {}'.format(H))
    self.logger.info('W       = {}'.format(W))
    self.logger.info('box     = {}'.format(m['num']))
    self.logger.info('classes = {}'.format(m['classes']))
    self.logger.info('scales  = {}'.format([sprob, sconf, snoob, scoor]))
    # Anchors logged as a list of ordered pairs for readability
    self.logger.info('anchors = {}'.format(list(zip(*[iter(anchors)]*2))))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.compat.v1.placeholder(tf.compat.v1.float32, size1)
    _confs = tf.compat.v1.placeholder(tf.compat.v1.float32, size2)
    _coord = tf.compat.v1.placeholder(tf.compat.v1.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.compat.v1.placeholder(tf.compat.v1.float32, size1)
    # material calculating IOU
    _areas = tf.compat.v1.placeholder(tf.compat.v1.float32, size2)
    _upleft = tf.compat.v1.placeholder(tf.compat.v1.float32, size2 + [2])
    _botright = tf.compat.v1.placeholder(tf.compat.v1.float32, size2 + [2])

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'areas': _areas, 'upleft': _upleft, 'botright': _botright
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.compat.v1.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.compat.v1.reshape(coords, [-1, H*W, B, 4])
    adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
    adjusted_coords_wh = tf.compat.v1.sqrt(tf.compat.v1.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.compat.v1.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.compat.v1.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.compat.v1.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    adjusted_prob = tf.compat.v1.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.compat.v1.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

    wh = tf.compat.v1.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    floor = centers - (wh * .5)
    ceil  = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft   = tf.compat.v1.maximum(floor, _upleft)
    intersect_botright = tf.compat.v1.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.compat.v1.maximum(intersect_wh, 0.0)
    intersect = tf.compat.v1.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.compat.v1.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.compat.v1.equal(iou, tf.compat.v1.reduce_max(iou, [2], True))
    best_box = tf.compat.v1.cast(best_box, tf.compat.v1.float32)
    confs = tf.compat.v1.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.compat.v1.concat(4 * [tf.compat.v1.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.compat.v1.concat(C * [tf.compat.v1.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.compat.v1.concat([_coord, tf.compat.v1.expand_dims(confs, 3), _probs], 3)
    wght = tf.compat.v1.concat([cooid, tf.compat.v1.expand_dims(conid, 3), proid], 3)

    self.logger.info('Building {} loss'.format(m['model']))
    loss = tf.compat.v1.pow(adjusted_net_out - true, 2)
    loss = tf.compat.v1.multiply(loss, wght)
    loss = tf.compat.v1.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.compat.v1.reduce_sum(loss, 1)
    self.loss = .5 * tf.compat.v1.reduce_mean(loss)
    tf.compat.v1.summary.scalar("/".join([os.path.basename(m['model']),
                                self.flags.trainer,
                                "loss"]), self.loss)
