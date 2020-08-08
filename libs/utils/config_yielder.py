import pickle
from typing import Generator
from libs.backend.io.darknet_config_file import DarknetConfigFile


class ConfigYielder(DarknetConfigFile):
    def __init__(self, model):
        super(ConfigYielder, self).__init__(model)
        self.h, self.w, self.c = self.metadata['inp_size']
        self.l = self.h * self.w * self.c
        self.flat = False
        self.conv = '.conv.' in model

    def yield_layers(self) -> Generator[dict, list, None]:
        yield self.metadata
        for i, section in enumerate(self.layers):
            layer_handler = self.get_layer_handler(section, i)
            try:
                yield [layer for layer in layer_handler(section, i)][0]
            except TypeError:
                raise TypeError('Layer {} not implemented'.format(section['type']))
            section['_size'] = list([self.h, self.w, self.c, self.l, self.flat])
        if not self.flat:
            self.metadata['out_size'] = [self.h, self.w, self.c]
        else:
            self.metadata['out_size'] = self.l

    def get_layer_handler(self, section: dict, i):
        handler_name = self._fixup_name(section['type'])
        handler = getattr(self, handler_name, [lambda section: None, lambda i: None])
        return handler

    @staticmethod
    def _pad(dimension, padding, size, stride):
        return (dimension + 2 * padding - size) // stride + 1

    @staticmethod
    def _local_pad(dimension, padding, size, stride):
        return (dimension - 1 - (1 - padding) * (size - 1)) // stride + 1

    @staticmethod
    def _fixup_name(section_name: str, snake_case=True):
        name = section_name.strip('[]')
        if snake_case:
            name = name.replace('-', '_')
        return name

    @staticmethod
    def _load_profile(file):
        with open(file, 'rb') as f:
            profiles = pickle.load(f, encoding='latin1')[0]
        return profiles

    @staticmethod
    def _list_keep(inp):
        return [int(x) for x in inp.split(',')]

    def select(self, section, i):
        if not self.flat:
            yield ['flatten', i]
            self.flat = True
        inp = section.get('input', None)
        if type(inp) is str:
            file = inp.split(',')[0]
            layer_num = int(inp.split(',')[1])
            profiles = self._load_profile(section['profile'])
            layer = profiles[layer_num]
        else:
            layer = inp
        activation = section.get('activation', 'logistic')
        section['keep'] = section['keep'].split('/')
        classes = int(section['keep'][-1])
        keep = self._list_keep(section['keep'][0])
        keep_n = len(keep)
        train_from = classes * section['bins']
        for count in range(section['bins'] - 1):
            for num in keep[-keep_n:]:
                keep += [num + classes]
        k = 1
        while self.layers[i - k]['type'] not in ['[connected]', '[extract]']:
            k += 1
            if i - k < 0:
                break
        if i - k < 0:
            l_ = self.l
        elif self.layers[i - k]['type'] == 'connected':
            l_ = self.layers[i - k]['output']
        else:
            l_ = self.layers[i - k].get('old', [self.l])[-1]
        yield ['select', i, l_, section['old_output'], activation, layer,
               section['output'], keep, train_from]
        if activation != 'linear':
            yield [activation, i]
        self.l = section['output']

    def convolutional(self, section, i):
        n = section.get('filters', 1)
        size = section.get('size', 1)
        stride = section.get('stride', 1)
        pad = section.get('pad', 0)
        padding = section.get('padding', 0)
        if pad:
            padding = size // 2
        activation = section.get('activation', 'logistic')
        batch_norm = section.get('batch_normalize', 0) or self.conv
        yield [self._fixup_name(section['type']), i, size, self.c, n, stride, padding,
               batch_norm, activation]
        if activation != 'linear':
            yield [activation, i]
        w_ = self._pad(self.w, padding, size, stride)
        h_ = self._pad(self.h, padding, size, stride)
        self.w, self.h, self.c = w_, h_, n
        self.l = self.w * self.h * self.c

    def crop(self, section, i):
        yield [self._fixup_name(section['type']), i]

    def local(self, section, i):
        n = section.get('filters', 1)
        size = section.get('size', 1)
        stride = section.get('stride', 1)
        pad = section.get('pad', 0)
        activation = section.get('activation', 'logistic')
        w_ = self._local_pad(self.w, pad, size, stride)
        h_ = self._local_pad(self.w, pad, size, stride)
        yield ['local', i, size, self.c, n, stride, pad, w_, h_, activation]
        if activation != 'linear':
            yield [activation, i]
        self.w, self.h, self.c = w_, h_, n
        self.l = self.w * self.h * self.c

    def conv_extract(self, section, i):
        profiles = self._load_profile(section['profile'])
        inp_layer = None
        inp = section['input']
        out = section['output']
        inp_layer = None
        if inp >= 0:
            inp_layer = profiles[inp]
        if inp_layer is not None:
            assert len(inp_layer) == self.c, \
                'Conv-extract does not match input dimension'
        out_layer = profiles[out]

        n = section.get('filters', 1)
        size = section.get('size', 1)
        stride = section.get('stride', 1)
        pad = section.get('pad', 0)
        padding = section.get('padding', 0)
        if pad:
            padding = size // 2
        activation = section.get('activation', 'logistic')
        batch_norm = section.get('batch_normalize', 0) or self.conv

        k = 1
        find = ['[convolutional]', '[conv-extract]']
        while self.layers[i - k]['type'] not in find:
            k += 1
            if i - k < 0:
                break
        if i - k >= 0:
            previous_layer = self.layers[i - k]
            c_ = previous_layer['filters']
        else:
            c_ = self.c

        yield [self._fixup_name(section['type'], snake_case=False), i, size, c_, n,
               stride, padding, batch_norm, activation, inp_layer, out_layer]
        if activation != 'linear':
            yield [activation, i]
        w_ = self._pad(self.w, padding, size, stride)
        h_ = self._pad(self.h, padding, size, stride)
        self.w, self.h, self.c = w_, h_, len(out_layer)
        self.l = self.w * self.h * self.c

    def conv_select(self, section, i):
        n = section.get('filters', 1)
        size = section.get('size', 1)
        stride = section.get('stride', 1)
        pad = section.get('pad', 0)
        padding = section.get('padding', 0)
        if pad:
            padding = size // 2
        activation = section.get('activation', 'logistic')
        batch_norm = section.get('batch_normalize', 0) or self.conv
        section['keep'] = section['keep'].split('/')
        classes = int(section['keep'][-1])
        keep = self._list_keep(section['keep'][0])

        segment = classes + 5
        assert n % segment == 0, \
            'conv-select: segment failed'
        bins = n // segment
        keep_idx = list()
        for j in range(bins):
            offset = j * segment
            for k in range(5):
                keep_idx += [offset + k]
            for k in keep:
                keep_idx += [offset + 5 + k]
        w_ = self._pad(self.w, padding, size, stride)
        h_ = self._pad(self.h, padding, size, stride)
        c_ = len(keep_idx)
        yield [self._fixup_name(section['type'], snake_case=False), i, size, self.c, n,
               stride, padding, batch_norm, activation, keep_idx, c_]
        self.w, self.h, self.c = w_, h_, c_
        self.l = self.w * self.h * self.c

    def maxpool(self, section, i):
        stride = section.get('stride', 1)
        size = section.get('size', stride)
        padding = section.get('padding', (size - 1) // 2)
        yield [self._fixup_name(section['type']), i, size, stride, padding]
        w_ = (self.w + 2 * padding) // section['stride']
        h_ = (self.h + 2 * padding) // section['stride']
        self.w, self.h = w_, h_
        self.l = self.w * self.h * self.c

    def connected(self, section, i):
        if not self.flat:
            yield ['flatten', i]
            self.flat = True
        activation = section.get('activation', 'logistic')
        yield ['connected', i, self.l, section['output'], activation]
        if activation != 'linear':
            yield [activation, i]
        self.l = section['output']

    def softmax(self, section, i):
        yield [self._fixup_name(section['type']), i, section['groups']]

    def extract(self, section, i):
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
            yield ['flatten', i]
            self.flat = True
        activation = section.get('activation', 'logistic')
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
        yield [self._fixup_name(section['type']), i] + old + [activation] + [inp_layer, out_layer]
        if activation != 'linear':
            yield [activation, i]
        self.l = len(out_layer)

    def route(self, section, i):
        routes = section['layers']
        if type(routes) is int:
            routes = [routes]
        else:
            routes = [int(x.strip()) for x in routes.split(',')]
        routes = [i + x if x < 0 else x for x in routes]
        for j, x in enumerate(routes):
            lx = self.layers[x]
            xtype = lx['type']
            _size = lx['_size'][:3]
            if j == 0:
                self.h, self.w, self.c = _size
            else:
                h_, w_, c_ = _size
                assert w_ == self.w and h_ ==  self.h, \
                    'Routing incompatible conv sizes'
                self.c += c_
        yield ['route', i, routes]
        self.l = self.w * self.h * self.c

    def shortcut(self, section, i):
        index = int(section['from'])
        activation = section.get('activation', 'logistic')
        assert activation == 'linear', \
            'Layer {} can only use linear activation'.format(section['type'])
        from_layer = self.layers[index]
        yield ['shortcut', i, from_layer]
        self.l = self.w * self.h * self.c

    def upsample(self, section, i):
        stride = section.get('stride', 1)
        assert stride == 2, \
            'Layer {} can only be of stride 2'.format(section['type'])
        w = self.w * stride
        h = self.h * stride
        yield [self._fixup_name(section['type']), i, stride, h, w]
        self.l = self.w * self.h * self.c

    def reorg(self, section, i):
        stride = section.get('stride', 1)
        yield [self._fixup_name(section['type']), i, stride]
        self.w = self.w // stride
        self.h = self.h // stride
        self.c = self.c * (stride ** 2)
        self.l = self.w * self.h * self.c

    def dropout(self, section, i):
        yield [self._fixup_name(section['type']), i, section['probability']]

    def avgpool(self, section, i):
        self.flat = True
        self.l = self.c
        yield [self._fixup_name(section['type']), i]
