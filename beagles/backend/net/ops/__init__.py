from beagles.backend.net.ops.simple import *
from beagles.backend.net.ops.convolution import *

op_types = {
    'convolutional': Convolutional,
    'connected': Connected,
    'maxpool': MaxPool,
    'stair': Stair,
    'hardtan': HardTan,
    'relu': Relu,
    'elu': Elu,
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
