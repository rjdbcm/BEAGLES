from beagles.backend.net.ops_v1.baseop import BaseOp
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell, ConvRNN2D
from tensorflow.keras.layers import BatchNormalization, SimpleRNN, GRU
from tensorflow.keras.layers import LSTM as _LSTM
import tensorflow as tf


class ConvLSTM(BaseOp):
    def call(self, inputs, **kwargs):
        kwargs.update({
            'filters': self.lay.output,
            'kernel_size': self.lay.size,
            'strides': (self.lay.stride, ) * 2,
            'activation': self.lay.activation,
            'padding': 'VALID'
        })
        pad = [[self.lay.pad, self.lay.pad]] * 2
        cell = PeepholeLSTMCell(**kwargs) if self.lay.peephole else ConvLSTM2DCell(**kwargs)
        temp = tf.pad(inputs, [[0, 0]] + pad + [[0, 0]])
        return ConvRNN2D(cell)(temp)


class LSTM(BaseOp):
    def call(self, inputs, **kwargs):
        num = self.lay.num_cells
        norm = self.lay.batch_norm
        temp = _LSTM(units=num)(inputs)
        return BatchNormalization(scale=False, center=False)(temp) if norm else temp


class RNN(BaseOp):
    def call(self, inputs, **kwargs):
        num = self.lay.num_cells
        norm = self.lay.batch_norm
        act = self.lay.activation
        temp = SimpleRNN(num, activation=act)(inputs)
        return BatchNormalization(scale=False, center=False)(temp) if norm else temp


class GRU(BaseOp):
    def call(self, inputs, **kwargs):
        num = self.lay.num_cells
        norm = self.lay.batch_norm
        temp = GRU(num)(inputs)
        return BatchNormalization(scale=False, center=False)(temp) if norm else temp
