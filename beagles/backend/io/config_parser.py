import pickle
import sys
from typing import Generator
from itertools import product
from beagles.backend.io.darknet_config_file import DarknetConfigFile
from beagles.base import SubsystemPrototype, Subsystem, register_subsystem


NAME, DEFAULT = range(2)
FILTERS = ('filters', 1)
SIZE = ('size', 1)
STRIDE = ('stride', 1)
PAD = ('pad', 0)
PADDING = ('padding', 0)
INPUT = ('input', None)
BATCHNORM = ('batch_normalize', 0)
ACTIVATION = 'activation'
OUTPUT = 'output'
FLATTEN = 'flatten'
CONNECTED = 'connected'
TYPE = 'type'
INP_SIZE = 'inp_size'
OUT_SIZE = 'out_size'
KEEP = 'keep'
BINS = 'bins'
LINEAR = 'linear'
SELECTABLE_LAY = ['[connected]', '[extract]']
EXTRACTABLE_LAY = ['[convolutional]', '[conv-extract]']
KEEP_DELIMITER = '/'

def _fix_name(section_name: str, snake_case=True, prefix: str = ''):
    name = section_name.strip('[]')
    if snake_case:
        name = name.replace('-', '_')
    name = prefix + name
    return name

def _yield_activation(activation, i):
    if activation != LINEAR:
        yield [activation, i]

def _pad(dimension, padding, size, stride):
    return (dimension + 2 * padding - size) // stride + 1

def _local_pad(dimension, padding, size, stride):
    return (dimension - 1 - (1 - padding) * (size - 1)) // stride + 1

def _load_profile(file):
    with open(file, 'rb') as f:
        profiles = pickle.load(f, encoding='latin1')[0]
    return profiles

def _list_keep(inp):
    return [int(x) for x in inp.split(',')]


class ConfigParser(SubsystemPrototype):
    def __init__(self, create_key, *args, **kwargs):
        super(ConfigParser, self).__init__(create_key, *args, **kwargs)

    @classmethod
    def create(cls, model):
        config = DarknetConfigFile(model)
        cls.layers, cls.metadata = config.tokens
        cls.h, cls.w, cls.c = cls.metadata[INP_SIZE]
        cls.l = cls.h * cls.w * cls.c
        cls.flat = False
        cls.conv = '.conv.' in model
        yield cls.metadata
        for i, section in enumerate(cls.layers):
            layer_handler = cls.get_register().get(_fix_name(section[TYPE]))
            handler = layer_handler(cls.create_key, cls)
            try:
                yield [layer for layer in handler(section, i)][0]
            except TypeError:
                raise TypeError('Layer {} not implemented'.format(section[TYPE]))
            section['_size'] = list([cls.h, cls.w, cls.c, cls.l, cls.flat])
        if not cls.flat:
            cls.metadata[OUT_SIZE] = [cls.h, cls.w, cls.c]
        else:
            cls.metadata[OUT_SIZE] = cls.l


@register_subsystem('', ConfigParser)
class DarknetConfigLayer(Subsystem):
    def constructor(self, parser):
        self.parser = parser
        self.layer_name, *_ = self.token.keys()

    def seek(self, layers, i):
        k = 1
        while self.parser.layers[i - k][TYPE] not in layers:
            k += 1
            if i - k < 0:
                break
        return k

    def _get_conv_properties(self, section):
        n = section.get(*FILTERS)
        size = section.get(*SIZE)
        stride = section.get(*STRIDE)
        pad = section.get(*PAD)
        padding = size // 2 if pad else section.get(*PADDING)
        activation = section.get(ACTIVATION)
        batch_norm = section.get(*BATCHNORM) or self.conv
        return [n, size, stride, padding, batch_norm, activation]


@register_subsystem('select', ConfigParser)
class Select(DarknetConfigLayer):

    constructor = DarknetConfigLayer.constructor
    seek = DarknetConfigLayer.seek

    def __call__(self, section, i):
        p = self.parser
        if not p.flat:
            yield [FLATTEN, i]
            p.flat = True
        inp = section.get(*INPUT)
        layer = self.profile(inp, section)
        activation = section.get(ACTIVATION)
        section[KEEP] = section[KEEP].split(KEEP_DELIMITER)
        classes = int(section[KEEP][-1])
        keep = _list_keep(section[KEEP][0])
        keep_n = len(keep)
        train_from = classes * section[BINS]
        for count in range(section[BINS] - 1):
            keep += [num + classes for num in keep[-keep_n:]]
        l_ = self.select_inps(i)
        yield [self.layer_name, i, l_, section['old_output'], activation, layer,
               section['output'], keep, train_from]
        _yield_activation(activation, i)
        p.l = section['output']

    def profile(self, inp, section):
        if type(inp) is str:
            file = inp.split(',')[0]
            layer_num = int(inp.split(',')[1])
            profiles = _load_profile(section['profile'])
            layer = profiles[layer_num]
        else:
            layer = inp
        return layer

    def select_inps(self, i):
        k = self.seek(SELECTABLE_LAY, i)
        if i - k < 0:
            l_ = self.parser.l
        elif self.parser.layers[i - k][TYPE] == CONNECTED:
            l_ = self.parser.layers[i - k]['output']
        else:
            l_ = self.parser.layers[i - k].get('old', [self.parser.l])[-1]
        return l_


@register_subsystem('convolutional', ConfigParser)
class Convolutional(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor
    _get_section_defaults = DarknetConfigLayer._get_conv_properties


    def __call__(self, section, i):
        p = self.parser
        n, size, stride, padding, batch_norm, activation = self._get_section_defaults(section)
        yield [self.layer_name, i, size, p.c, n, stride, padding,
               batch_norm, activation]
        _yield_activation(activation, i)
        w_ = _pad(p.w, padding, size, stride)
        h_ = _pad(p.h, padding, size, stride)
        p.w, p.h, p.c = w_, h_, n
        p.l = p.w * p.h * p.c

@register_subsystem('crop', ConfigParser)
class Crop(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        yield [self.layer_name, i]

@register_subsystem('local', ConfigParser)
class Local(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor
    _get_section_defaults = DarknetConfigLayer._get_conv_properties

    def __call__(self, section, i):
        p = self.parser
        n, size, stride, *_, activation = self._get_section_defaults(section)
        pad = section.get(*PAD)
        w_ = _local_pad(self.w, pad, size, stride)
        h_ = _local_pad(self.w, pad, size, stride)
        yield [self.layer_name, i, size, p.c, n, stride, pad, w_, h_,
               activation]
        _yield_activation(activation, i)
        p.w, p.h, p.c = w_, h_, n
        p.l = p.w * p.h * p.c

@register_subsystem('conv-extract', ConfigParser)
class ConvExtract(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor
    seek = DarknetConfigLayer.seek
    _get_section_defaults = DarknetConfigLayer._get_conv_properties

    def __call__(self, section, i):
        p = self.parser
        profiles = _load_profile(section['profile'])
        inp_layer = None
        inp = section[INPUT[NAME]]
        out = section['output']
        inp_layer = None
        if inp >= 0:
            inp_layer = profiles[inp]
        if inp_layer is not None:
            assert len(inp_layer) == p.c, 'Conv-extract does not match input dimension'
        out_layer = profiles[out]
        n, size, stride, padding, batch_norm, activation = self._get_section_defaults(section)
        c_ = self.extract_channels(i)
        yield [self.layer_name, i, size, c_, n,
               stride, padding, batch_norm, activation, inp_layer, out_layer]
        _yield_activation(activation, i)
        w_ = _pad(p.w, padding, size, stride)
        h_ = _pad(p.h, padding, size, stride)
        p.w, p.h, p.c = w_, h_, len(out_layer)
        p.l = p.w * p.h * p.c

    def extract_channels(self, i):
        k = self.seek(EXTRACTABLE_LAY, i)
        if i - k >= 0:
            previous_layer = self.parser.layers[i - k]
            c_ = previous_layer['filters']
        else:
            c_ = self.parser.c
        return c_



@register_subsystem('conv-select', ConfigParser)
class ConvSelect(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor
    _get_section_defaults = DarknetConfigLayer._get_conv_properties

    def __call__(self, section, i):
        p = self.parser
        n, size, stride, padding, *mess = self._get_section_defaults(section)
        section[KEEP] = section[KEEP].split(KEEP_DELIMITER)
        classes = int(section[KEEP][-1])
        keep = _list_keep(section[KEEP][0])
        segment = classes + 5
        assert n % segment == 0, 'conv-select: segment failed'
        bins = n // segment
        keep_idx = list()
        for j in range(bins):
            offset = j * segment
            keep_idx.append([offset + k for k in range(5)])
            keep_idx.append([offset + 5 + k for k in keep])
        w_ = _pad(p.w, padding, size, stride)
        h_ = _pad(p.h, padding, size, stride)
        c_ = len(keep_idx)
        yield [self.layer_name, i, size, p.c, n, stride, padding, *mess, keep_idx, c_]
        p.w, p.h, p.c = w_, h_, c_
        p.l = p.w * p.h * p.c


@register_subsystem('maxpool', ConfigParser)
class MaxPool(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        p = self.parser
        stride = section.get(*STRIDE)
        size = section.get(SIZE[0], stride)
        padding = section.get(PADDING[0], (size - 1) // 2)
        yield [self.layer_name, i, size, stride, padding]
        w_ = (p.w + 2 * padding) // section[STRIDE[0]]
        h_ = (p.h + 2 * padding) // section[STRIDE[0]]
        p.w, p.h = w_, h_
        p.l = p.w * p.h * p.c


@register_subsystem('connected', ConfigParser)
class Connected(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        p = self.parser
        if not p.flat:
            yield [FLATTEN, i]
            p.flat = True
        activation = section.get(ACTIVATION)
        yield [self.layer_name, i, p.l, section['output'], activation]
        _yield_activation(activation, i)
        p.l = section['output']

@register_subsystem('extract', ConfigParser)
class Extract(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor
    def new(self, input_layer, heights, widths):
        range_c = range(self.parser.c)
        range_h = range(self.parser.h)
        range_w = range(self.parser.w)
        new_inp = list()
        for i in range_c:
            if i not in input_layer:
                continue
            new_inp += [k + widths * (j + heights * i) for j in range_h for k in range_w]
        return new_inp
    def __call__(self, section, i):
        p = self.parser
        if not p.flat:
            yield [FLATTEN, i]
            p.flat = True
        activation = section.get(ACTIVATION)
        profiles = _load_profile(section['profile'])
        inp_layer = None
        inp = section[INPUT[NAME]]
        out = section['output']
        if inp >= 0:
            inp_layer = profiles[inp]
        out_layer = profiles[out]
        old = section['old']
        old = [int(x) for x in old.split(',')]
        if inp_layer is not None:
            if len(old) > 2:
                h_, w_, c_, n_ = old
                inp_layer = self.new(inp_layer, h_, w_)
                old = [h_ * w_ * c_, n_]
            if len(inp_layer) == p.l:
                msg = f'Extract does not match input dimension {len(inp_layer)} != {p.l}.'
                raise ValueError(msg)
        section['old'] = old
        yield [self.layer_name, i, *old, activation, inp_layer, out_layer]
        _yield_activation(activation, i)
        p.l = len(out_layer)

@register_subsystem('route', ConfigParser)
class Route(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        p = self.parser
        routes = section['layers']
        if type(routes) is int:
            routes = [routes]
        else:
            routes = [int(x.strip()) for x in routes.split(',')]
        routes = [i + x if x < 0 else x for x in routes]
        for j, x in enumerate(routes):
            lx = p.layers[x]
            _size = lx['_size'][:3]
            if j == 0:
                p.h, p.w, p.c = _size
            else:
                h_, w_, c_ = _size
                assert w_ == p.w and h_ == p.h, \
                    f'Routing incompatible sizes from {lx[TYPE]}({h_}x{w_}x{c_})'
                p.c += c_
        yield [self.layer_name, i, routes]
        p.l = p.w * p.h * p.c

@register_subsystem('shortcut', ConfigParser)
class Shortcut(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        p = self.parser
        index = int(section['from'])
        activation = section.get(ACTIVATION)
        assert activation == LINEAR, 'Layer {} can only use linear activation'.format(
            section[TYPE])
        from_layer = p.layers[index]
        yield [self.layer_name, i, from_layer]
        p.l = p.w * p.h * p.c

@register_subsystem('upsample', ConfigParser)
class Upsample(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        p = self.parser
        stride = section.get(*STRIDE)
        assert stride == 2, \
            'Layer {} can only be of stride 2'.format(section[TYPE])
        w = p.w * stride
        h = p.h * stride
        yield [self.layer_name, i, stride, h, w]
        p.l = p.w * p.h * p.c

@register_subsystem(token='reorg', prototype=ConfigParser)
class Reorg(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        p = self.parser
        stride = section.get(*STRIDE)
        yield [self.layer_name, i, stride]
        p.w = p.w // stride
        p.h = p.h // stride
        p.c = p.c * (stride ** 2)
        p.l = p.w * p.h * p.c

@register_subsystem('avgpool', ConfigParser)
class AvgPool(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        self.parser.flat = True
        self.parser.l = self.parser.c
        yield [self.layer_name, i]

@register_subsystem('dropout', ConfigParser)
class Dropout(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        yield [self.layer_name, i, section['probability']]

@register_subsystem('softmax', ConfigParser)
class SoftMax(DarknetConfigLayer):
    constructor = DarknetConfigLayer.constructor

    def __call__(self, section, i):
        yield [self.layer_name, i, section['groups']]