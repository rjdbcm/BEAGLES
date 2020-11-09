from beagles.backend.net.ops.baseop import BaseOp
import tensorflow as tf
import numpy as np

class Reorg(BaseOp):
    def __init__(self, *args):
        super(Reorg, self).__init__(*args)

    def call(self, inputs, **kwargs):
        s = self.lay.stride
        return tf.image.extract_patches(inputs, [1,s,s,1], [1,s,s,1], [1,1,1,1], 'VALID')


class Local(BaseOp):
    def __init__(self, *args):
        super(Local, self).__init__(*args)

    def build(self, input_shape):
        ksz = (self.lay.ksize,) * 2
        filt = self.lay.wshape['kernels'][-1]
        stride = (self.lay.stride,) * 2
        self._lay = tf.keras.layers.LocallyConnected2D(filt, ksz, stride,
                                                       trainable=True, name=self.scope)

    def call(self, inputs, **kwargs):
        pad = [[self.lay.pad, self.lay.pad]] * 2
        temp = tf.pad(inputs, [[0, 0]] + pad + [[0, 0]])
        return self._lay(temp)


class Convolutional(BaseOp):
    def __init__(self, *args, **kwargs):
        super(Convolutional,self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=tuple(self.lay.wshape['biases']),
            initializer="random_normal",
            trainable=True,
            name=f'{self.scope}-bias'
        )

    def call(self, inputs, **kwargs):
        pad = [[self.lay.pad, self.lay.pad]] * 2
        temp = tf.pad(inputs, [[0, 0]] + pad + [[0, 0]])
        self.kw = self.add_weight(shape=tuple(self.lay.wshape['kernel']), dtype=tf.float32,
                                  name=f'{self.scope}-kweight' )
        temp = tf.nn.conv2d(temp, self.kw, padding='VALID',
                            name=self.scope, strides=[1] + [self.lay.stride] * 2 + [1])
        if self.lay.batch_norm:
            temp = self.batchnorm(temp)
        return tf.nn.bias_add(temp, self.b)

    def batchnorm(self, inputs):
        if not self.var:
            temp = (inputs - self.lay.w['moving_mean'])
            temp /= (np.sqrt(self.lay.w['moving_variance']) + 1e-5)
            temp *= self.lay.w['gamma']
            return temp
        else:
            args = dict({
                'center': False,
                'scale': True,
                'epsilon': 1e-5,
                'name': self.scope,
            })
            return tf.keras.layers.BatchNormalization(**args)(inputs)
