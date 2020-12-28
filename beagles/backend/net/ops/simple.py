from beagles.backend.net.ops.baseop import BaseOp
from tensorflow.python.framework.errors_impl import UnimplementedError
import tensorflow as tf
import sys


class Route(BaseOp):
    def __init__(self, *args):
        super(Route, self).__init__(*args)

    def call(self, inputs, **kwargs):
        routes = self.lay.routes
        routes_out = list()
        _this = None
        for r in routes:
            this = self.inp
            while this.lay.number != r:
                _this = this
                this = this.inp
                assert this is not None, f'Routing to non-existence {r}'
            try:
                routes_out += [this(inputs)]
            except UnimplementedError as e:
                raise e
        return tf.keras.layers.Concatenate(3)(routes_out)


class Connected(BaseOp):
    def __init__(self, *args):
        super(Connected, self).__init__(*args)

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


class Flatten(BaseOp):
    def __init__(self, *args):
        super(Flatten, self).__init__(*args)

    def call(self, inputs, **kwargs):
        return tf.keras.layers.Flatten(tf.transpose(inputs, [0, 3, 1, 2]), name=self.scope)


class SoftMax(BaseOp):
    def __init__(self, *args):
        super(SoftMax, self).__init__(*args)

    def call(self, inputs, **kwargs):
        return tf.nn.softmax(inputs)


class AvgPool(BaseOp):
    def __init__(self, *args):
        super(AvgPool, self).__init__(*args)

    def call(self, inputs, **kwargs):
        return tf.reduce_mean(inputs, [1,2], name=self.scope)


class DropOut(BaseOp):
    def __init__(self, *args):
        super(DropOut, self).__init__(*args)

    def call(self, inputs, **kwargs):
        if self.lay.h['pdrop'] is None:
            self.lay.h['pdrop'] = 0.0
        return tf.nn.dropout(inputs, self.lay.h['pdrop'], name=self.scope)


class Crop(BaseOp):
    def __init__(self, *args):
        super(Crop, self).__init__(*args)

    def call(self, inputs, **kwargs):
        return inputs * 2.0 - 1.0


class MaxPool(BaseOp):
    def __init__(self, *args):
        super(MaxPool, self).__init__(*args)

    def call(self, inputs, **kwargs):
        return tf.nn.max_pool(inputs, padding='SAME',
                              ksize=[1] + [self.lay.ksize] * 2 + [1],
                              strides=[1] + [self.lay.stride] * 2 + [1], name=self.scope)


class Shortcut(BaseOp):
    def __init__(self, *args):
        super(Shortcut, self).__init__(*args)

    def call(self, inputs, **kwargs):
        from_layer = self.lay.from_layer
        this = self.inp
        # walk backwards thru inputs until we reach the target from_layer
        while this.lay.number != from_layer:
            this = this.inp
            assert this is not None, f'Shortcut to non-existence {self.lay.from_layer}'
        return tf.add(inputs, this, name=self.scope)


class UpSample(BaseOp):
    def __init__(self, *args):
        super(UpSample, self).__init__(*args)

    def call(self, inputs, **kwargs):
        size = (self.lay.height, self.lay.width)
        tf.image.resize(inputs, size, method='nearest', name=self.scope)


# ---Activations---
class Stair(BaseOp):
    def call(self, inputs, **kwargs):
        n = tf.floor(inputs)
        f = tf.floor(tf.divide(inputs, 2), name=self.scope)
        return f if n % 2 == 0 else tf.add(tf.subtract(inputs, n), f, name=self.scope)


class HardTan(BaseOp):
    def call(self, inputs, **kwargs):
        t = tf.shape(self.inp.out)
        cond = tf.less(inputs, tf.ones(t)) and tf.greater(inputs, tf.negative(tf.ones(t)))
        return tf.where(cond, tf.ones(t), tf.zeros(t), name=self.scope)


class Relu(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.nn.relu(inputs, name=self.scope)


class Gelu(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.identity(0.5 * inputs *(1 + tf.tanh(0.797885 * inputs + 0.035677 * tf.pow(inputs, 3))), name=self.scope)


class Selu(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.nn.selu(inputs, name=self.scope)


class Tanh(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.nn.tanh(inputs, name=self.scope)


class Logistic(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.identity(1.0/(1.0 + tf.exp(-inputs)), name=self.scope)


class Loggy(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.identity(2.0/(1.0 + tf.exp(-inputs)) - 1, name=self.scope)


class Relie(BaseOp):
    def call(self, inputs, **kwargs):
        false = tf.identity(.01*inputs, name=self.scope)
        true = tf.identity(inputs, name=self.scope)
        return tf.cond(tf.greater(inputs, 0), true, false)


class PSLE(BaseOp):
    def call(self, inputs, **kwargs):
        default = .125 * inputs + .5
        lt_value = tf.cond(tf.less(inputs, -4), .01 * (inputs + 4), default)
        gt_value = tf.cond(tf.greater(inputs, 4), .01 * (inputs -4) + 1, default)
        if lt_value == gt_value:
            return default
        elif lt_value != default:
            return lt_value
        elif gt_value != default:
            return gt_value


class LHTan(BaseOp):
    def call(self, inputs, **kwargs):
        lt_value = tf.cond(tf.less(inputs, 0), .001 * inputs, inputs)
        gt_value = tf.cond(tf.greater(inputs, 1), .001 * (inputs - 1), inputs)
        if lt_value == gt_value:
            return tf.identity(inputs, name=self.scope)
        elif lt_value != inputs:
            return lt_value
        elif gt_value != inputs:
            return gt_value


class Linear(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.identity(inputs, name=self.scope)


class Elu(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.nn.elu(inputs, name=self.scope)


class Leaky(BaseOp):
    def call(self, inputs, **kwargs):
        return tf.maximum(.1 * inputs, inputs, name=self.scope)
