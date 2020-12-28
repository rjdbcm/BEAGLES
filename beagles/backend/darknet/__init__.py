# noinspection PyUnresolvedReferences
import os
from beagles.backend.net.ops import op_create
from beagles.backend.darknet.layer import Layer
from beagles.backend.darknet.convolution import *
from beagles.backend.darknet.connected import *
from beagles.backend.darknet.rnn import *
from beagles.backend.io.config_parser import ConfigParser
from beagles.io.flags import SharedFlagIO
from beagles.base.constants import CFG_EXT, WGT_EXT


class avgpool_layer(Layer):
    def setup(self):
        """Not Implemented"""
        pass

    def finalize(self, *args):
        """Not Implemented"""
        pass


class crop_layer(Layer):
    def setup(self):
        """Not Implemented"""
        pass

    def finalize(self, *args):
        """Not Implemented"""
        pass


# noinspection PyAttributeOutsideInit
class upsample_layer(Layer):
    def setup(self, stride, h, w):
        self.stride = stride
        self.height = h
        self.width = w

    def finalize(self, *args):
        """Not Implemented"""
        pass


# noinspection PyAttributeOutsideInit
class shortcut_layer(Layer):
    def setup(self, from_layer):
        self.from_layer = from_layer

    def finalize(self, *args):
        """Not Implemented"""
        pass


# noinspection PyAttributeOutsideInit
class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

    def finalize(self, *args):
        """Not Implemented"""
        pass


# noinspection PyAttributeOutsideInit
class softmax_layer(Layer):
    def setup(self, groups):
        self.groups = groups

    def finalize(self, *args):
        """Not Implemented"""
        pass


class dropout_layer(Layer):
    def setup(self, p):
        self.h['pdrop'] = p

    def finalize(self, *args):
        """Not Implemented"""
        pass


# noinspection PyAttributeOutsideInit
class route_layer(Layer):
    def setup(self, routes):
        self.routes = routes

    def finalize(self, *args):
        """Not Implemented"""
        pass


# noinspection PyAttributeOutsideInit
class reorg_layer(Layer):
    def setup(self, stride):
        self.stride = stride

    def finalize(self, *args):
        """Not Implemented"""
        pass


darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'shortcut': shortcut_layer,
    'upsample': upsample_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer,
    'crop': crop_layer,
    'local': local_layer,
    'select': select_layer,
    'route': route_layer,
    'reorg': reorg_layer,
    'conv-select': conv_select_layer,
    'conv-extract': conv_extract_layer,
    'extract': extract_layer,
    'lstm': lstm_layer,
    'rnn': rnn_layer,
    'gru': gru_layer
}
""":obj:`dict`: darknet layer types

   Items:                                                   
         'dropout': :obj:`beagles.backend.darknet.dropout_layer`

         'connected': :obj:`beagles.backend.darknet.connected_layer`

         'maxpool': :obj:`beagles.backend.darknet.maxpool_layer`

         'shortcut': :obj:`beagles.backend.darknet.shortcut_layer`

         'upsample': :obj:`beagles.backend.darknet.upsample_layer`

         'convolutional': :obj:`beagles.backend.darknet.convolutional_layer`

         'avgpool': :obj:`beagles.backend.darknet.avgpool_layer`

         'softmax': :obj:`beagles.backend.darknet.softmax_layer`

         'crop': :obj:`beagles.backend.darknet.crop_layer`

         'local': :obj:`beagles.backend.darknet.local_layer`

         'select': :obj:`beagles.backend.darknet.select_layer`

         'route': :obj:`beagles.backend.darknet.route_layer`

         'reorg': :obj:`beagles.backend.darknet.reorg_layer`

         'conv-select': :obj:`beagles.backend.darknet.conv_select_layer`

         'conv-extract': :obj:`beagles.backend.darknet.conv_extract_layer`

         'extract': :obj:`beagles.backend.darknet.extract_layer`

"""


def create_darkop(ltype: str, num: int, *args):
    op_class = darkops.get(ltype, Layer)
    return op_class(ltype, num, *args)
"""Darknet operations factory method.

    Args:
        ltype: layer type one of :obj:`beagles.backend.darknet.darkops` keys.

        num: numerical index of layer

        *args: variable list of layer characteristics yielded by :meth:`beagles.backend.io.ConfigParser.parse_layers`

"""


class Darknet(object):

    def __init__(self, flags):
        self.io = SharedFlagIO(subprogram=True)
        self.get_weight_src(flags)
        self.modify = False
        self.io.log.info('Parsing {}'.format(self.src_cfg))
        src_parsed = self.create_ops()
        self.meta, self.layers = src_parsed

    def compile(self):
        num_layer = n_train = len(self.layers) or 0
        layers = list()
        roof = num_layer - n_train
        prev = None
        for i, lay in enumerate(self.layers):
            lay = op_create(lay, prev, i, roof)
            layers.append(lay)
            prev = lay
        return layers

    def get_weight_src(self, flags):
        """
        analyse flags.load to know where is the
        source binary and what is its config.
        can be: None, flags.model, or some other
        """
        self.src_bin = flags.model + WGT_EXT
        self.src_bin = flags.binary + self.src_bin
        self.src_bin = os.path.abspath(self.src_bin)
        exist = os.path.isfile(self.src_bin)

        if flags.load == str():
            flags.load = int()
        if type(flags.load) is int:
            self.src_cfg = flags.model
            if flags.load:
                self.src_bin = None
            elif not exist:
                self.src_bin = None
        else:
            self.src_bin = flags.load
            name = self.model_name(flags.load)
            cfg_path = os.path.join(flags.config, name + CFG_EXT)
            if not os.path.isfile(cfg_path):
                self.io.logger.warn(f'{cfg_path} not found, use {flags.model} instead')
                cfg_path = flags.model
            self.src_cfg = cfg_path
            flags.load = int()

    @staticmethod
    def model_name(file_path):
        file_name = os.path.basename(file_path)
        ext = str()
        if '.' in file_name:  # exclude extension
            file_name = file_name.split('.')
            ext = file_name[-1]
            file_name = '.'.join(file_name[:-1])
        if ext == str() or ext == 'meta':  # ckpt file
            file_name = file_name.split('-')
            num = int(file_name[-1])
            return '-'.join(file_name[:-1])
        if ext == 'weights':
            return file_name

    def create_ops(self):
        """
        return a list of `layers` objects (darkop.py)
        given path to binaries/ and configs/
        """
        cfg_layers = ConfigParser.create(self.src_cfg)

        meta = dict()
        layers = list()
        try:
            for i, info in enumerate(cfg_layers):
                if i == 0:
                    meta = info
                    continue
                else:
                    new = create_darkop(*info)
                layers.append(new)
        except TypeError as e:
            self.io.flags.error = str(e) + str(cfg_layers)
            self.io.log.error(str(e))
            self.io.send_flags()
            raise
        return meta, layers

Layer = Layer
""":class:`Layer`"""

dropout_layer = dropout_layer
""":class:`dropout_layer`"""

connected_layer = connected_layer
""":class:`connected_layer`"""

maxpool_layer = maxpool_layer
""":class:`maxpool_layer`"""

shortcut_layer = shortcut_layer
""":class:`shortcut_layer`"""

upsample_layer = upsample_layer
""":class:`upsample_layer`"""

convolutional_layer = convolutional_layer
""":class:`convolutional_layer`"""

avgpool_layer = avgpool_layer
""":class:`avgpool_layer`"""

softmax_layer = softmax_layer
""":class:`softmax_layer`"""

crop_layer = crop_layer
""":class:`crop_layer`"""

local_layer = local_layer
""":class:`local_layer`"""

select_layer = select_layer
""":class:`select_layer`"""

route_layer = route_layer
""":class:`route_layer`"""

reorg_layer = reorg_layer
""":class:`reorg_layer`"""

conv_select_layer = conv_select_layer
""":class:`conv_select_layer`"""

conv_extract_layer = conv_extract_layer
""":class:`conv_extract_layer`"""

extract_layer = extract_layer
""":class:`extract_layer`"""