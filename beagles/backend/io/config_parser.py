import pickle
import sys
from typing import Generator
from itertools import product
from beagles.backend.io.darknet_config_file import DarknetConfigFile


ACTIVATION = 'activation'
FILTERS = ('filters', 1)
SIZE = ('size', 1)
STRIDE = ('stride', 1)
PAD = ('pad', 0)
PADDING = ('padding', 0)
INPUT = ('input', None)
BATCHNORM = ('batch_normalize', 0)
FLATTEN = 'flatten'
TYPE = 'type'
INP_SIZE = 'inp_size'
OUT_SIZE = 'out_size'
KEEP = 'keep'
BINS = 'bins'
LINEAR = 'linear'
SELECTABLE_LAY = ['[connected]', '[extract]']
EXTRACTABLE_LAY = ['[convolutional]', '[conv-extract]']
KEEP_DELIMITER = '/'


class ConfigParser:
    def __init__(self, model):
        config = DarknetConfigFile(model)
        self.layers, self.metadata = config.tokens
        self.h, self.w, self.c = self.metadata[INP_SIZE]
        self.l = self.h * self.w * self.c
        self.flat = False
        self.conv = '.conv.' in model

    def parse_layers(self) -> Generator[dict, list, None]:
        """Generator used to create Layer subclass *args from darknet .cfg files.

        Yields:

            0: metadata

            1 ... N: *args for Layer subclass returned by :meth:`beagles.backend. darknet.darknet.create_darkop` for layers 1 to N
        """

        yield self.metadata
        for i, section in enumerate(self.layers):
            layer_handler = self._get_layer_handler(section, i)
            try:
                yield [layer for layer in layer_handler(section, i)][0]
            except TypeError:
                raise TypeError('Layer {} not implemented'.format(section[TYPE]))
            section['_size'] = list([self.h, self.w, self.c, self.l, self.flat])
        if not self.flat:
            self.metadata[OUT_SIZE] = [self.h, self.w, self.c]
        else:
            self.metadata[OUT_SIZE] = self.l

    def _get_layer_handler(self, section: dict, i):
        handler_name = _fix_name(section[TYPE], prefix='_')
        handler = getattr(self, handler_name, [lambda section: str, lambda i: int])
        return handler
    
    def _get_section_defaults(self, section):
        n = section.get(*FILTERS)
        size = section.get(*SIZE)
        stride = section.get(*STRIDE)
        pad = section.get(*PAD)
        padding = size // 2 if pad else section.get(*PADDING)
        activation = section.get(ACTIVATION)
        batch_norm = section.get(*BATCHNORM) or self.conv
        return [n, size, stride, padding, batch_norm, activation]

    @staticmethod
    def _pad(dimension, padding, size, stride):
        return (dimension + 2 * padding - size) // stride + 1

    @staticmethod
    def _local_pad(dimension, padding, size, stride):
        return (dimension - 1 - (1 - padding) * (size - 1)) // stride + 1

    @staticmethod
    def _load_profile(file):
        with open(file, 'rb') as f:
            profiles = pickle.load(f, encoding='latin1')[0]
        return profiles

    @staticmethod
    def _list_keep(inp):
        return [int(x) for x in inp.split(',')]

    def _select(self, section, i):
        if not self.flat:
            yield [FLATTEN, i]
            self.flat = True
        inp = section.get(*INPUT)
        if type(inp) is str:
            file = inp.split(',')[0]
            layer_num = int(inp.split(',')[1])
            profiles = self._load_profile(section['profile'])
            layer = profiles[layer_num]
        else:
            layer = inp
        activation = section.get(ACTIVATION)
        section[KEEP] = section[KEEP].split(KEEP_DELIMITER)
        classes = int(section[KEEP][-1])
        keep = self._list_keep(section[KEEP][0])
        keep_n = len(keep)
        train_from = classes * section[BINS]
        for count in range(section[BINS] - 1):
            for num in keep[-keep_n:]:
                keep += [num + classes]
        k = 1
        while self.layers[i - k][TYPE] not in SELECTABLE_LAY:
            k += 1
            if i - k < 0:
                break
        if i - k < 0:
            l_ = self.l
        elif self.layers[i - k][TYPE] == 'connected':
            l_ = self.layers[i - k]['output']
        else:
            l_ = self.layers[i - k].get('old', [self.l])[-1]
        yield [_fix_name(section[TYPE]), i, l_, section['old_output'], activation, layer,
               section['output'], keep, train_from]
        if activation != LINEAR:
            yield [activation, i]
        self.l = section['output']

    def _convolutional(self, section, i):
        n, size, stride, padding, batch_norm, activation = self._get_section_defaults(
            section)
        yield [_fix_name(section[TYPE]), i, size, self.c, n, stride, padding,
               batch_norm, activation]
        if activation != LINEAR:
            yield [activation, i]
        w_ = self._pad(self.w, padding, size, stride)
        h_ = self._pad(self.h, padding, size, stride)
        self.w, self.h, self.c = w_, h_, n
        self.l = self.w * self.h * self.c

    def _crop(self, section, i):
        yield [_fix_name(section[TYPE]), i]

    def _local(self, section, i):
        n, size, stride, *_, activation = self._get_section_defaults(section)
        pad = section.get(*PAD)
        w_ = self._local_pad(self.w, pad, size, stride)
        h_ = self._local_pad(self.w, pad, size, stride)
        yield [_fix_name(section[TYPE]), i, size, self.c, n, stride, pad, w_, h_, activation]
        if activation != LINEAR:
            yield [activation, i]
        self.w, self.h, self.c = w_, h_, n
        self.l = self.w * self.h * self.c

    def _conv_extract(self, section, i):
        profiles = self._load_profile(section['profile'])
        inp_layer = None
        inp = section['input']
        out = section['output']
        inp_layer = None
        if inp >= 0:
            inp_layer = profiles[inp]
        if inp_layer is not None:
            assert len(inp_layer) == self.c, 'Conv-extract does not match input dimension'
        out_layer = profiles[out]
        n, size, stride, padding, batch_norm, activation  = self._get_section_defaults(
            section)
        k = 1
        while self.layers[i - k][TYPE] not in EXTRACTABLE_LAY:
            k += 1
            if i - k < 0:
                break
        if i - k >= 0:
            previous_layer = self.layers[i - k]
            c_ = previous_layer['filters']
        else:
            c_ = self.c

        yield [_fix_name(section[TYPE], snake_case=False), i, size, c_, n,
               stride, padding, batch_norm, activation, inp_layer, out_layer]
        if activation != LINEAR:
            yield [activation, i]
        w_ = self._pad(self.w, padding, size, stride)
        h_ = self._pad(self.h, padding, size, stride)
        self.w, self.h, self.c = w_, h_, len(out_layer)
        self.l = self.w * self.h * self.c

    def _conv_select(self, section, i):
        n, size, stride, padding, *mess = self._get_section_defaults(section)
        section[KEEP] = section[KEEP].split(KEEP_DELIMITER)
        classes = int(section[KEEP][-1])
        keep = self._list_keep(section[KEEP][0])
        segment = classes + 5
        assert n % segment == 0, 'conv-select: segment failed'
        bins = n // segment
        keep_idx = list()
        for j in range(bins):
            offset = j * segment
            keep_idx.append([offset + k for k in range(5)])
            keep_idx.append([offset + 5 + k for k in keep])
        w_ = self._pad(self.w, padding, size, stride)
        h_ = self._pad(self.h, padding, size, stride)
        c_ = len(keep_idx)
        name = _fix_name(section[TYPE], snake_case=False)
        yield [name, i, size, self.c, n, stride, padding, *mess, keep_idx, c_]
        self.w, self.h, self.c = w_, h_, c_
        self.l = self.w * self.h * self.c

    def _maxpool(self, section, i):
        stride = section.get(*STRIDE)
        size = section.get(SIZE[0], stride)
        padding = section.get(PADDING[0], (size - 1) // 2)
        yield [_fix_name(section[TYPE]), i, size, stride, padding]
        w_ = (self.w + 2 * padding) // section[STRIDE[0]]
        h_ = (self.h + 2 * padding) // section[STRIDE[0]]
        self.w, self.h = w_, h_
        self.l = self.w * self.h * self.c

    def _connected(self, section, i):
        if not self.flat:
            yield [FLATTEN, i]
            self.flat = True
        activation = section.get(ACTIVATION)
        yield ['connected', i, self.l, section['output'], activation]
        if activation != LINEAR:
            yield [activation, i]
        self.l = section['output']

    def _softmax(self, section, i):
        yield [_fix_name(section[TYPE]), i, section['groups']]

    def _extract(self, section, i):
        def new_input_layer(input_layer, colors: list, heights: list, widths: list):
            new_inp = list()
            for p in range(colors[1]):
                for q in range(heights[1]):
                    for r in range(widths[1]):
                        if p not in input_layer:
                            continue
                        new_inp += [r + widths[0] * (q + heights[0] * p)]
            return new_inp
        if not self.flat:
            yield [FLATTEN, i]
            self.flat = True
        activation = section.get(ACTIVATION)
        profiles = self._load_profile(section['profile'])
        inp_layer = None
        inp = section['input']
        out = section['output']
        if inp >= 0:
            inp_layer = profiles[inp]
        out_layer = profiles[out]
        old = section['old']
        old = [int(x) for x in old.split(',')]
        if inp_layer is not None:
            if len(old) > 2:
                h_, w_, c_, n_ = old
                inp_layer = new_input_layer(inp_layer, [c_, self.c], [h_, self.h], [w_, self.w])
                old = [h_ * w_ * c_, n_]
            assert len(inp_layer) == self.l, 'Extract does not match input dimension {} =/= {}'.format(len(inp_layer), self.l)
        section['old'] = old
        yield [_fix_name(section[TYPE]), i] + old + [activation] + [inp_layer, out_layer]
        if activation != LINEAR:
            yield [activation, i]
        self.l = len(out_layer)

    def _route(self, section, i):
        routes = section['layers']
        if type(routes) is int:
            routes = [routes]
        else:
            routes = [int(x.strip()) for x in routes.split(',')]
        routes = [i + x if x < 0 else x for x in routes]
        for j, x in enumerate(routes):
            lx = self.layers[x]
            xtype = lx[TYPE]
            _size = lx['_size'][:3]
            if j == 0:
                self.h, self.w, self.c = _size
            else:
                h_, w_, c_ = _size
                assert w_ == self.w and h_ ==  self.h, \
                    'Routing incompatible conv sizes'
                self.c += c_
        yield [_fix_name(section[TYPE]), i, routes]
        self.l = self.w * self.h * self.c

    def _shortcut(self, section, i):
        index = int(section['from'])
        activation = section.get(ACTIVATION)
        assert activation == LINEAR, 'Layer {} can only use linear activation'.format(section[TYPE])
        from_layer = self.layers[index]
        yield [_fix_name(section[TYPE]), i, from_layer]
        self.l = self.w * self.h * self.c

    def _upsample(self, section, i):
        stride = section.get(*STRIDE)
        assert stride == 2, \
            'Layer {} can only be of stride 2'.format(section[TYPE])
        w = self.w * stride
        h = self.h * stride
        yield [_fix_name(section[TYPE]), i, stride, h, w]
        self.l = self.w * self.h * self.c

    def _reorg(self, section, i):
        stride = section.get(*STRIDE)
        yield [_fix_name(section[TYPE]), i, stride]
        self.w = self.w // stride
        self.h = self.h // stride
        self.c = self.c * (stride ** 2)
        self.l = self.w * self.h * self.c

    def _dropout(self, section, i):
        yield [_fix_name(section[TYPE]), i, section['probability']]

    def _avgpool(self, section, i):
        self.flat = True
        self.l = self.c
        yield [_fix_name(section[TYPE]), i]


def _fix_name(section_name: str, snake_case=True, prefix: str = ''):
    name = section_name.strip('[]')
    if snake_case:
        name = name.replace('-', '_')
    name = prefix + name
    return name