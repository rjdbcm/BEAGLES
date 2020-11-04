from beagles.backend.net.ops.baseop import BaseOp
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell, ConvRNN2D
from tensorflow.keras.layers import BatchNormalization, LSTM, SimpleRNN, GRU
import tensorflow as tf

class conv_lstm(BaseOp):
    def forward(self):
        kwargs = {
            'filters': self.lay.output,
            'kernel_size': self.lay.size,
            'strides': (self.lay.stride, ) * 2,
            'activation': self.lay.activation,
            'padding': 'VALID'
        }
        pad = [[self.lay.pad, self.lay.pad]] * 2
        cell = PeepholeLSTMCell(**kwargs) if self.lay.peephole else ConvLSTM2DCell(**kwargs)
        nn = ConvRNN2D(cell)
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
        self.out = nn(temp)

    def speak(self):
        size = self.lay.size
        filters = self.lay.output
        norm = self.lay.batch_norm * "+bnorm"
        pad = self.lay.pad
        stride = self.lay.stride
        peep = self.lay.peephole * "+peep"
        activation = self.lay.activation
        msg = f'conv-lstm {size}x{size}p{pad}_{stride} {norm} {peep} {activation}'


class lstm(BaseOp):
    def forward(self):
        num = self.lay.num_cells
        norm = self.lay.batch_norm
        inp = self.inp.out
        temp = LSTM(units=num)(inp)
        self.out = BatchNormalization(scale=False, center=False)(temp) if norm else temp

    def speak(self):
        msg = f'lstm {self.lay.num_cells} {"+bnorm" * self.lay.batch_norm}'
        return msg

class rnn(BaseOp):
    def forward(self):
        num = self.lay.num_cells
        norm = self.lay.batch_norm
        act = self.lay.activation
        inp = self.inp.out
        temp = SimpleRNN(num, activation=act)(inp)
        self.out = BatchNormalization(scale=False, center=False)(temp) if norm else temp

    def speak(self):
        msg = f'rnn {self.lay.num_cells}x{self.lay.num_cells} {"+bnorm" if self.lay.batch_norm else ""}'
        return msg

class gru(BaseOp):
    def forward(self):
        inp = self.inp.out
        num = self.lay.num_cells
        norm = self.lay.batch_norm
        temp = GRU(num)(inp)
        self.out = BatchNormalization(scale=False, center=False)(temp) if norm else temp

    def speak(self):
        msg = f'gru {self.lay.num_cells} {"+bnorm" if self.lay.batch_norm else ""}'
        return msg
