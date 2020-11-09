from beagles.backend.net.ops.simple import *
from beagles.backend.net.ops.rnn import *
from beagles.backend.net.ops.convolution import *

op_types = {
    'convolutional': Convolutional,
    'connected': Connected,
    'conv-lstm': ConvLSTM,
    'lstm': LSTM,
    'rnn': RNN,
    'gru': GRU,
    'maxpool': MaxPool,
    'stair': Stair,
    'hardtan': HardTan,
    'relu': Relu,
    'relie': Relie,
    'linear': Linear,
    'logistic': Logistic,
    'loggy': Loggy,
    'gelu': Gelu,
    'selu': Selu,
    'elu': Elu,
    'psle': PSLE,
    'tanh': Tanh,
    'lhtan': LHTan,
    'leaky': Leaky,
    'shortcut': Shortcut,
    'upsample': UpSample,
    'dropout': DropOut,
    'flatten': Flatten,
    'avgpool': AvgPool,
    'softmax': SoftMax,
    'crop': Crop,
    'local': Local,
    'route': Route,
    'reorg': Reorg,
}


def op_create(*args):
    layer_type = list(args)[0].type
    return op_types[layer_type](*args)
