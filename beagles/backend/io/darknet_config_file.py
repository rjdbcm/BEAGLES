import os
import json
from beagles.base.errors import DarknetConfigEmpty


class DarknetConfigFile:
    """Tokenize darknet style .cfg file to layers and metadata"""

    def __init__(self, path):
        self.config_file = path
        file_ext = os.path.splitext(self.config_file)[1]

        if file_ext == '.cfg' and os.path.exists(path):
            self.layers, self.metadata = self.__tokenize()
        elif file_ext == '.json' and os.path.exists(path):
            self.from_json(path)
        else:
            raise FileNotFoundError(path)

        if not len(self):
            raise DarknetConfigEmpty(self.config_file)

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)

    def __eq__(self, other):
        """compare the layers of the config file"""
        lay = self.layers
        lay_ = other.layers
        return lay_ == lay

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        """naively combine layers and metadata"""
        self.metadata = dict(self.metadata, **other.metadata)
        self.layers = self.layers + other.layers
        return self

    def to_json(self, **kwargs):
        name = os.path.splitext(self.config_file)[0] + '.json'
        data = json.dumps(self.__dict__, **kwargs)
        with open(name, 'w') as f:
            f.write(data)
        return name

    def from_json(self, file):
        with open(file, 'r') as f:
            self.__dict__.update(json.load(f))
        return self

    @property
    def tokens(self):
        return self.layers, self.metadata

    def __tokenize(self):
        def split_on_eq(line, i=1):
            return line.split('=')[i].strip()

        with open(self.config_file, 'rb') as f:
            lines = f.readlines()
        lines = [line.decode() for line in lines]
        meta = dict()
        layers = list()
        h, w, c = [int()] * 3
        layer = dict()
        error = False
        for line in lines:
            line = line.strip()
            line = line.split('#')[0]
            if '[' in line:
                if layer != dict():
                    if layer['type'] == '[net]':
                        h = layer['height']
                        w = layer['width']
                        c = layer['channels']
                        meta['net'] = layer
                        try:
                            meta['net']['augment'] = [str(i.strip(' ')) for i in meta['net']['augment'].split(',')]
                        except KeyError:
                            pass
                    else:
                        if layer['type'] == '[crop]':
                            h = layer['crop_height']
                            w = layer['crop_width']
                        layers += [layer]
                layer = {'type': line}
            else:
                try:
                    i = float(split_on_eq(line))
                    if i == int(i):
                        i = int(i)
                    layer[line.split('=')[0].strip()] = i
                except (IndexError, ValueError):
                    try:
                        key = split_on_eq(line, 0)
                        val = split_on_eq(line, 1)
                        layer[key] = val
                    except IndexError:
                        pass
        meta.update(layer)  # last layer contains meta info
        if 'anchors' in meta:
            splits = meta['anchors'].split(',')
            anchors = [float(x.strip()) for x in splits]
            meta['anchors'] = anchors
        meta['model'] = self.config_file  # path to cfg, not model name
        meta['inp_size'] = [h, w, c]
        return layers, meta