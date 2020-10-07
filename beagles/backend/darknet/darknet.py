import os
import time
import sys
from beagles.backend.darknet.darkop import *
from beagles.backend.io.config_parser import ConfigParser
from beagles.io.flags import SharedFlagIO
from beagles.backend.io.loader import Loader
from beagles.base.constants import CFG_EXT, WGT_EXT

darkops = darkops
""":obj:`dict`: darknet layer types
   
   Items:                                                   
         'dropout': :obj:`beagles.backend. darknet.darknet.dropout_layer`
        
         'connected': :obj:`beagles.backend. darknet.darknet.connected_layer`
        
         'maxpool': :obj:`beagles.backend. darknet.darknet.maxpool_layer`
        
         'shortcut': :obj:`beagles.backend. darknet.darknet.shortcut_layer`
        
         'upsample': :obj:`beagles.backend. darknet.darknet.upsample_layer`
        
         'convolutional': :obj:`beagles.backend. darknet.darknet.convolutional_layer`
        
         'avgpool': :obj:`beagles.backend. darknet.darknet.avgpool_layer`
        
         'softmax': :obj:`beagles.backend. darknet.darknet.softmax_layer`
        
         'crop': :obj:`beagles.backend. darknet.darknet.crop_layer`
        
         'local': :obj:`beagles.backend. darknet.darknet.local_layer`
        
         'select': :obj:`beagles.backend. darknet.darknet.select_layer`
        
         'route': :obj:`beagles.backend. darknet.darknet.route_layer`
        
         'reorg': :obj:`beagles.backend. darknet.darknet.reorg_layer`
        
         'conv-select': :obj:`beagles.backend. darknet.darknet.conv_select_layer`
        
         'conv-extract': :obj:`beagles.backend.darknet.darkop.conv_extract_layer`
        
         'extract': :obj:`beagles.backend.darknet.darkop.extract_layer`

"""

create_darkop = create_darkop
"""Darknet operations factory method.

    Args:
        ltype: layer type one of :obj:`beagles.backend. darknet.darknet.darkops` keys.
        
        num: numerical index of layer
        
        *args: variable list of layer characteristics yielded by :meth:`beagles.backend.io.ConfigParser.parse_layers`

"""


class Darknet(object):

    def __init__(self, flags):
        self.io = SharedFlagIO(subprogram=True)
        self.get_weight_src(flags)
        self.modify = False
        self.io.logger.info('Parsing {}'.format(self.src_cfg))
        src_parsed = self.create_ops()
        self.meta, self.layers = src_parsed
        self.load_weights()

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
            self.io.flags.error = str(e)
            self.io.logger.error(str(e))
            self.io.send_flags()
            raise
        return meta, layers

    def load_weights(self):
        """
        Use `layers` and Loader to load .weights file
        """
        self.io.logger.info(f'Loading {self.src_bin} ...')
        start = time.time()

        args = [self.src_bin, self.layers]
        wgts_loader = Loader.create(*args)
        for layer in self.layers:
            layer.load(wgts_loader)

        stop = time.time()
        self.io.logger.info('Finished in {}s'.format(stop - start))

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