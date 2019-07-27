from ..utils.process import cfg_yielder
from .darkop import create_darkop
from ..utils import loader
from ..utils.flags import FlagIO
import warnings
import time
import os


class Darknet(FlagIO, object):
    _EXT = '.weights'

    def __init__(self, flags):
        FlagIO.__init__(self, subprogram=True)
        self.get_weight_src(flags)
        self.modify = False

        self.logger.info('Parsing {}'.format(self.src_cfg))
        src_parsed = self.parse_cfg(self.src_cfg, flags)
        self.src_meta, self.src_layers = src_parsed

        if self.src_cfg == flags.model:
            self.meta, self.layers = src_parsed
        else:
            self.logger.info('Parsing {}'.format(flags.model))
            des_parsed = self.parse_cfg(flags.model, flags)
            self.meta, self.layers = des_parsed

        self.load_weights()

    def get_weight_src(self, flags):
        """
        analyse flags.load to know where is the
        source binary and what is its config.
        can be: None, flags.model, or some other
        """
        self.src_bin = flags.model + self._EXT
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
            assert os.path.isfile(flags.load), '{} not found'.format(flags.load)
            self.src_bin = flags.load
            name = loader.model_name(flags.load)
            cfg_path = os.path.join(flags.config, name + '.cfg')
            if not os.path.isfile(cfg_path):
                warnings.warn(
                    '{} not found, use {} instead'.format(
                        cfg_path, flags.model))
                cfg_path = flags.model
            self.src_cfg = cfg_path
            flags.load = int()

    def parse_cfg(self, model, flags):
        """
        return a list of `layers` objects (darkop.py)
        given path to binaries/ and configs/
        """
        args = [model, flags.binary]
        cfg_layers = cfg_yielder(*args)
        meta = dict();
        layers = list()
        for i, info in enumerate(cfg_layers):
            if i == 0:
                meta = info; continue
            else:
                new = create_darkop(*info)
            layers.append(new)
        return meta, layers

    def load_weights(self):
        """
        Use `layers` and Loader to load .weights file
        """
        self.logger.info('Loading {} ...'.format(self.src_bin))
        start = time.time()

        args = [self.src_bin, self.src_layers]
        wgts_loader = loader.create_loader(*args)
        for layer in self.layers:
            layer.load(wgts_loader)

        stop = time.time()
        self.logger.info('Finished in {}s'.format(stop - start))
