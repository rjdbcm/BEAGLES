class DarknetConfigFile:
    """Read the .cfg file to extract layers and metadata"""

    def __init__(self, path):

        self.config_file = path
        self.layers, self.metadata = self.__parse()

    def __parse(self):
        def split_on_eq(line, i=1):
            return line.split('=')[i].strip()

        with open(self.config_file, 'rb') as f:
            lines = f.readlines()
        lines = [line.decode() for line in lines]
        meta = dict()
        layers = list()
        h, w, c = [int()] * 3
        layer = dict()
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