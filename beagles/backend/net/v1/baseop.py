import numpy as np
import tensorflow as tf
from beagles.io.logs import get_logger

FORM = '{:>6} | {:>6} | {:<32} | {}'
FORM_ = '{}+{}+{}+{}'
LINE = FORM_.format('-' * 7, '-' * 8, '-' * 34, '-' * 15)
HEADER = FORM.format('Source', 'Train?', 'Layer description', 'Output size')


def _shape(tensor):  # work for both tf.Tensor & np.ndarray
    if type(tensor) in [tf.Variable, tf.Tensor]:
        return tensor.get_shape()
    else:
        return tensor.shape

def _name(tensor):
    try:
        name = tensor.name.split(':')[0]
    except AttributeError:
        name = 'input'
    return name


class BaseOp(tf.Module):
    """
    BaseOp objects initialise with a darknet's `layer` object
    and input tensor of that layer `inp`, it calculates the
    output of this layer and place the result in self.out
    """

    # let slim take care of the following vars
    _SLIM = ['gamma', 'moving_mean', 'moving_variance']

    def __init__(self, layer, inp, num, roof, feed):
        super(BaseOp, self).__init__()
        self.inp = inp  # BaseOp
        self.num = num  # int
        self.out = None  # tf.Tensor
        self.logger, self.logfile = get_logger()
        self.lay = layer
        self.scope = '{}-{}'.format(str(self.num), self.lay.type)
        self.gap = roof - self.num
        self.var = not self.gap > 0
        self.act = 'Load '
        self.convert(feed)
        if self.var:
            self.train_msg = 'Yes  '
        else:
            self.train_msg = 'No   '
        self.forward()

    def convert(self, feed):
        """convert self.lay to variables & placeholders"""
        for var in self.lay.wshape:
            self.wrap_variable(var)
        for ph in self.lay.h:
            self.wrap_pholder(ph, feed)

    def wrap_variable(self, var):
        """wrap layer.w into variables"""
        val = self.lay.w.get(var, None)
        if val is None:
            shape = self.lay.wshape[var]
            args = [0., 1e-2, shape]
            if 'moving_mean' in var:
                val = np.zeros(shape)
            elif 'moving_variance' in var:
                val = np.ones(shape)
            else:
                val = np.random.normal(*args)
            self.lay.w[var] = val.astype(np.float32)
            self.act = 'Init '
        if not self.var:
            return
        val = self.lay.w[var]
        if var in self._SLIM:
            return
        with self.name_scope:
            self.lay.w[var] = tf.Variable(val, shape=self.lay.wshape[var], dtype=tf.float32)

    def wrap_pholder(self, ph, feed):
        """wrap layer.h into placeholders"""
        phtype = type(self.lay.h[ph])
        if phtype is not dict:
            return

        sig = '{}/{}'.format(self.scope, ph)
        val = self.lay.h[ph]

        self.lay.h[ph] = tf.Variable(val['dfault'], val['shape'], name=sig)
        try:
            feed[self.lay.h[ph]] = val['feed']
        except TypeError:
            feed[self.lay.h[ph].experimental_ref()] = val['feed']

    def verbalise(self):  # console speaker
        msg = str()
        inp = _name(self.inp.out)
        if inp == 'input':
            #  imitating log format from utils.flags
            self.logger.info(FORM.format('', '', 'input', _shape(self.inp.out)))
        if not self.act:
            return msg
        self.logger.info(FORM.format(self.act, self.train_msg, self.speak(), _shape(self.out)))

    def forward(self):
        pass

    def speak(self):
        pass