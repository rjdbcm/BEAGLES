import tensorflow as tf
from tensorflow.keras.regularizers import L1L2, l1
from tensorflow.keras.losses import hinge
from beagles.io.flags import SharedFlagIO

_LOSS_TYPE = ['sse', 'l2', 'smooth', 'sparse', 'l1', 'softmax']


def loss(self, y_pred, y_true):
    losses = self.type.keys()
    loss_type = self.meta['type'].strip('[]')
    out_size = self.meta['out_size']
    H, W, _ = self.meta['inp_size']
    HW = H * W
    try:
        assert loss_type in losses, f'Loss type {loss_type} not implemented'
    except AssertionError as e:
        self.flags.error = str(e)
        self.logger.error(str(e))
        SharedFlagIO.send_flags(self)
        raise

    if self.first:
        self.logger.info('{} loss hyper-parameters:'.format(self.meta['model']))
        self.logger.info('Input Grid Size   = {}'.format(HW))
        self.logger.info('Number of Outputs = {}'.format(out_size))
        self.first = False

    diff = y_true - y_pred
    if loss_type in ['sse', '12']:
        return tf.nn.l2_loss(diff)
    elif loss_type == 'mse':
        return tf.keras.losses.MSE(y_true, y_pred)
    elif loss_type == ['smooth']:
        small = tf.cast(diff < 1, tf.float32)
        large = 1. - small
        return L1L2(tf.multiply(diff, large), tf.multiply(diff, small))
    elif loss_type in ['sparse', 'l1']:
        return l1(diff)
    elif loss_type == 'softmax':
        _loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred)
        return tf.reduce_mean(_loss)
    # elif loss_type == 'svm':
    #     assert 'train_size' in self.meta, 'Must specify'
    #     size = self.meta['train_size']
    #     self.nu = tf.Variable(tf.ones([self.flags.size, self.num_classes]))
    #     self.loss(hinge(self.nu))