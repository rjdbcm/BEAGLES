from keras.layers import Flatten
import tensorflow as tf
import os


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
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S # number of grid cells

    self.logger.info('{} loss hyper-parameters:'.format(m['model']))
    self.logger.info('side    = {}'.format(m['side']))
    self.logger.info('box     = {}'.format(m['num']))
    self.logger.info('classes = {}'.format(m['classes']))
    self.logger.info('scales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, SS, C]
    size2 = [None, SS, B]

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
    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.math.pow(coords[:, :, :, 2:4], 2) * S  # unit: grid cell
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]  # unit: grid cell^2
    centers = coords[:, :, :, 0:2]  # [batch, SS, B, 2]
    floor = centers - (wh * .5)  # [batch, SS, B, 2]
    ceil  = centers + (wh * .5)  # [batch, SS, B, 2]

    # calculate the intersection areas
    intersect_upleft   = tf.math.maximum(floor, _upleft)
    intersect_botright = tf.math.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.math.maximum(intersect_wh, 0.0)
    intersect = tf.math.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.math.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.math.equal(iou, tf.math.reduce_max(iou, [2], True))
    best_box = tf.cast(best_box, tf.float32)
    confs = tf.math.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    proid = sprob * _proid

    # flatten 'em all
    probs = Flatten(_probs)
    proid = Flatten(proid)
    confs = Flatten(confs)
    conid = Flatten(conid)
    coord = Flatten(_coord)
    cooid = Flatten(cooid)

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat([probs, confs, coord], 1)
    wght = tf.concat([proid, conid, cooid], 1)
    self.logger.info('Building {} loss'.format(m['model']))
    loss = tf.math.pow(net_out - true, 2)
    loss = tf.math.multiply(loss, wght)
    loss = tf.math.reduce_sum(loss, 1)
    self.loss = .5 * tf.math.reduce_mean(loss)
    tf.compat.v1.summary.scalar("/".join([os.path.basename(m['model']),
                                self.flags.trainer,
                                "loss"]), self.loss)