import tensorflow as tf


class BaseOp(tf.keras.Model):

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def __init__(self, layer, inp, num, roof):
        super(BaseOp, self).__init__()
        self.lay = layer
        self.inp = inp
        self.num = num
        self.gap = roof - self.num
        self.var = not self.gap > 0
        self.scope = '{}-{}'.format(str(self.num), self.lay.type)


