import tensorflow.compat.v1.layers as slim
from beagles.backend.net.ops.baseop import BaseOp, BaseOpV2
from deprecated.sphinx import deprecated
import tensorflow as tf


class route(BaseOp):
    def forward(self):
        routes = self.lay.routes
        routes_out = list()
        for r in routes:
            this = self.inp
            while this.lay.number != r:
                this = this.inp
                assert this is not None, \
                    'Routing to non-existence {}'.format(r)
            routes_out += [this.out]
        self.out = tf.concat(routes_out, 3)

    def speak(self):
        msg = 'concat {}'
        return msg.format(self.lay.routes)

class Connected(BaseOpV2):
    def __init__(self, layer):
        super(Connected, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=tuple(self.lay.wshape['weights']),
            initializer="random_normal",
            trainable=True,
            name=f'{self.scope}-weights'
        )
        self.b = self.add_weight(
            shape=tuple(self.lay.wshape['biases']),
            initializer="random_normal",
            trainable=True,
            name=f'{self.scope}-bias'
        )

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b

class connected(BaseOp):
    def forward(self):
        self.out = self.inp.out * self.lay.w['weights'] + self.lay.w['biases']

    def speak(self):
        layer = self.lay
        args = [layer.inp, layer.out]
        args += [layer.activation]
        msg = 'full {} x {}  {}'
        return msg.format(*args)


@deprecated(reason='DEPRECATION', version="1.0.0a1")
class select(connected):

    def speak(self):
        layer = self.lay
        args = [layer.inp, layer.out]
        args += [layer.activation]
        msg = 'sele {} x {}  {}'
        return msg.format(*args)


@deprecated(reason='DEPRECATION', version="1.0.0a1")
class extract(connected):

    def speak(self):
        layer = self.lay
        args = [len(layer.inp), len(layer.out)]
        args += [layer.activation]
        msg = 'extr {} x {}  {}'
        return msg.format(*args)


class flatten(BaseOp):
    def forward(self):
        temp = tf.transpose(self.inp.out, [0, 3, 1, 2])
        self.out = slim.flatten(temp, name=self.scope)

    def speak(self): return 'flat'


class softmax(BaseOp):
    def forward(self):
        self.out = tf.nn.softmax(self.inp.out)

    def speak(self): return 'softmax()'


class avgpool(BaseOp):
    def forward(self):
        self.out = tf.reduce_mean(
            self.inp.out, [1, 2],
            name=self.scope
        )

    def speak(self): return 'avgpool()'


class dropout(BaseOp):
    def forward(self):
        if self.lay.h['pdrop'] is None:
            self.lay.h['pdrop'] = 1.0
        self.out = tf.nn.dropout(
            self.inp.out,
            self.lay.h['pdrop'],
            name=self.scope
        )

    def speak(self): return 'drop'

class Crop(tf.keras.layers.Layer):
    def __init__(self):
        super(Crop, self).__init__()

    def call(self, inputs):
        return inputs * 2.0 - 1.0

class crop(BaseOp):
    def forward(self):
        self.out = self.inp.out * 2. - 1.

    def speak(self):
        return 'scale to (-1, 1)'

class MaxPool(BaseOpV2):
    def call(self, inputs, **kwargs):
        return tf.nn.max_pool(
            inputs, padding='SAME',
            ksize=[1] + [self.lay.ksize] * 2 + [1],
            strides=[1] + [self.lay.stride] * 2 + [1],
            name=self.scope
        )

class maxpool(BaseOp):
    def forward(self):
        self.out = tf.nn.max_pool(
            self.inp.out, padding='SAME',
            ksize=[1] + [self.lay.ksize] * 2 + [1],
            strides=[1] + [self.lay.stride] * 2 + [1],
            name=self.scope
        )

    def speak(self):
        l = self.lay
        return 'maxp {}x{}p{}_{}'.format(
            l.ksize, l.ksize, l.pad, l.stride)


class shortcut(BaseOp):
    def forward(self):
        from_layer = self.lay.from_layer
        this = self.inp
        # walk backwards thru inputs until we reach the target from_layer
        while this.lay.number != from_layer:
            this = this.inp
            assert this is not None, f'Shortcut to non-existence {self.lay.from_layer}'
        self.out = tf.add(self.inp.out, this.out, name=self.scope)

    def speak(self):
        l = self.lay
        return 'shortcut from {}'.format(l.from_layer)


class upsample(BaseOp):
    def forward(self):
        size = (self.lay.height, self.lay.width)
        self.out = tf.image.resize(self.inp.out, size, method='nearest', name=self.scope)

    def speak(self):
        return 'upsample {}'.format(self.lay.stride)


# ---Activations---
class stair(BaseOp):
    def forward(self):
        n = tf.floor(self.inp.out)
        f = tf.floor(tf.divide(self.inp.out, 2), name=self.scope)
        if n % 2 == 0:
            self.out = f
        else:
            self.out = tf.add(tf.subtract(self.inp.out, n), f, name=self.scope)

    def verbalise(self):
        pass


class hardtan(BaseOp):
    def forward(self):
        # self.out = 1 if x > -1 and x < 1 else 0
        t = tf.shape(self.inp.out)
        cond = tf.less(self.inp.out, tf.ones(t)) and tf.greater(self.inp.out, tf.negative(tf.ones(t)))
        self.out = tf.where(cond, tf.ones(t), tf.zeros(t), name=self.scope)

    def verbalise(self):
        pass


class relu(BaseOp):
    def forward(self):
        self.out = tf.nn.relu(self.inp.out, name=self.scope)

    def verbalise(self):
        pass


class elu(BaseOp):
    def forward(self):
        self.out = tf.nn.elu(self.inp.out, name=self.scope)

    def verbalise(self):
        pass


class leaky(BaseOp):
    def forward(self):
        self.out = tf.maximum(.1 * self.inp.out, self.inp.out, name=self.scope)

    def verbalise(self):
        pass

class Identity(BaseOpV2):
    def __init__(self, input):
        return input


class identity(BaseOp):
    def __init__(self, inp):
        self.inp = None
        self.out = inp
