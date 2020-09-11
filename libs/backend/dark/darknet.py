import os
import time
from libs.backend.dark.darkop import create_darkop
from libs.backend.io.config_parser import ConfigParser
from libs.io.flags import FlagIO
from libs.backend.io.loader import Loader
from libs.constants import CFG_EXT, WGT_EXT


class Darknet(object):

    def __init__(self, flags):
        self.io = FlagIO(subprogram=True)
        self.get_weight_src(flags)
        self.modify = False
        self.config = ConfigParser(self.src_cfg)

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
        cfg_layers = self.config.parse_layers()

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
