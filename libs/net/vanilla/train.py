import tensorflow as tf
from ...utils.flags import FlagIO

_LOSS_TYPE = ['sse', 'l2', 'smooth',
              'sparse', 'l1', 'softmax',
              'svm', 'fisher']


def loss(self, net_out):
    m = self.meta
    loss_type = m['type'].strip('[]')
    out_size = m['out_size']
    H, W, _ = m['inp_size']
    HW = H * W
    try:
        assert loss_type in _LOSS_TYPE, \
            'Loss type {} not implemented'.format(loss_type)
    except AssertionError as e:
        self.flags.error = str(e)
        self.logger.error(str(e))
        FlagIO.send_flags(self)
        raise

    self.logger.info('{} loss hyper-parameters:'.format(m['model']))
    self.logger.info('Input Grid Size   = {}'.format(HW))
    self.logger.info('Number of Outputs = {}'.format(out_size))

    out = net_out
    out_shape = out.get_shape()
    out_dtype = out.dtype.base_dtype
    _truth = tf.placeholder(out_dtype, out_shape)

    self.placeholders = dict({
        'truth': _truth
    })

    diff = _truth - out
    if loss_type in ['sse', '12']:
        self.loss = tf.nn.l2_loss(diff)

    elif loss_type == ['smooth']:
        small = tf.cast(diff < 1, tf.float32)
        large = 1. - small
        l1_loss = tf.nn.l1_loss(tf.multiply(diff, large))
        l2_loss = tf.nn.l2_loss(tf.multiply(diff, small))
        loss = l1_loss + l2_loss

    elif loss_type in ['sparse', 'l1']:
        loss = l1_loss(diff)

    elif loss_type == 'softmax':
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net_out)
        loss = tf.reduce_mean(loss)

    elif loss_type == 'svm':
        assert 'train_size' in m, \
            'Must specify'
        size = m['train_size']
        self.nu = tf.Variable(tf.ones([self.flags.size, num_classes]))
