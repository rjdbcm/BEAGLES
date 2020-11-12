from beagles.backend.net.v1.simple import *
from beagles.backend.net.v1.rnn import *
from beagles.backend.net.v1.convolution import *

op_types = {
    'convolutional': convolutional,
    'connected': connected,
    'conv-lstm': conv_lstm,
    'lstm': lstm,
    'rnn': rnn,
    'gru': gru,
    'maxpool': maxpool,
    'stair': stair,
    'hardtan': hardtan,
    'relu': relu,
    'elu': elu,
    'leaky': leaky,
    'shortcut': shortcut,
    'upsample': upsample,
    'dropout': dropout,
    'flatten': flatten,
    'avgpool': avgpool,
    'softmax': softmax,
    'crop': crop,
    'local': local,
    'route': route,
    'reorg': reorg,
}


def op_create(*args):
    layer_type = list(args)[0].type
    return op_types[layer_type](*args)
